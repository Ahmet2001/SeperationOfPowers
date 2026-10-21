"""Fine-tune one Mercan role on a training mix (PROJECT_MASTER §9.1).

    python training/train_role.py --role yasama \\
        --mix  /arf/scratch/egitimg16/nedolm_0p8b/tooluse_inspect/yasama-yurutme-yargi/mix/v2 \\
        --run-dir /arf/scratch/egitimg16/nedolm_0p8b/runs/mercan-yasama-<date>

Runs in the TRUBA training container next to the NedoLM package (``nedolm``)
and the NedoTokenizer Python package. Full-parameter training from the DPO
checkpoint (§9 base checkpoint); every record is serialized as
``nedolm.chat_format`` does (``<|mesaj_basla|>rol\\n...<|mesaj_bitir|>\\n``,
EOS after the conversation) and the loss covers only the last assistant
message and its ``<|mesaj_bitir|>``: earlier assistant turns are context.
Records are packed into rows split at EOS (FlashAttention varlen, positions
restart per conversation), so no compute goes to padding.

Validation reports the target loss and the exact-match rate per action:
the share of records whose every target token is the argmax given the
correct prefix, which is what greedy decoding would produce.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

MESSAGE_START_ID, MESSAGE_END_ID, EOS_ID, PAD_ID = 32000, 32001, 2, 0
CHAT_ROLES = {"system": "sistem", "user": "kullanici", "assistant": "asistan"}
TOKENIZER_SHA256 = "72412d981dac65a29d1767bc98821fc2bcffc2de53c534e7c719598515bfb600"

# Per-role defaults; every one can be overridden on the command line.
ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    # A classifier: one action name per record, forgetting general chat is harmless.
    "yasama": {"epochs": 2, "lr": 1e-5, "select_by": "exact_match"},
    # JSON arguments; exact match is the metric that matters.
    "yurutme": {"epochs": 2, "lr": 1e-5, "select_by": "exact_match"},
    # Writes to the user: a lower rate keeps the DPO model's Turkish.
    "yargi": {"epochs": 2, "lr": 5e-6, "select_by": "loss"},
}

# ------------------------------------------------------------- serialization


def encode_record(encode: Callable[[str], list[int]], messages: Sequence[dict[str, str]]):
    """Token ids of one conversation and the [start, end) span the loss covers
    (the last message's body and <|mesaj_bitir|>)."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("a training record ends with the assistant target")
    ids: list[int] = []
    newline = encode("\n")
    for index, message in enumerate(messages):
        ids += [MESSAGE_START_ID, *encode(CHAT_ROLES[message["role"]] + "\n")]
        body = encode(message["content"])
        if index == len(messages) - 1:
            start = len(ids)
            ids += [*body, MESSAGE_END_ID]
            end = len(ids)
            ids += newline
        else:
            ids += [*body, MESSAGE_END_ID, *newline]
    ids.append(EOS_ID)
    return ids, start, end


def pack(lengths: Sequence[int], order: Sequence[int], capacity: int) -> list[list[int]]:
    """Fill rows of ``capacity`` tokens with whole records in the given order."""
    rows: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index in order:
        length = lengths[index]
        if length > capacity:
            raise ValueError(f"record {index} has {length} tokens, row capacity is {capacity}")
        if used + length > capacity:
            rows.append(current)
            current, used = [], 0
        current.append(index)
        used += length
    if current:
        rows.append(current)
    return rows


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float) -> float:
    """Linear warmup, then cosine decay to ``peak * min_ratio``."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------- data


class Corpus:
    """One split, tokenized: a flat uint16 token array plus per-record spans."""

    def __init__(self, tokens, offsets, lengths, target_starts, target_ends, actions):
        self.tokens, self.offsets, self.lengths = tokens, offsets, lengths
        self.target_starts, self.target_ends, self.actions = target_starts, target_ends, actions

    def __len__(self) -> int:
        return len(self.lengths)

    @property
    def target_tokens(self) -> int:
        return int((self.target_ends - self.target_starts).sum())

    @classmethod
    def build(cls, path: Path, encode: Callable[[str], list[int]]) -> "Corpus":
        import numpy as np

        chunks, lengths, starts, ends, actions = [], [], [], [], []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                ids, start, end = encode_record(encode, record["messages"])
                chunks.append(np.asarray(ids, dtype=np.uint16))
                lengths.append(len(ids))
                starts.append(start)
                ends.append(end)
                actions.append(record["action"])
        lengths_array = np.asarray(lengths, dtype=np.int64)
        offsets = np.zeros(len(lengths), dtype=np.int64)
        np.cumsum(lengths_array[:-1], out=offsets[1:])
        return cls(np.concatenate(chunks), offsets, lengths_array,
                   np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64), actions)

    def save(self, path: Path) -> None:
        import numpy as np

        np.savez(path, tokens=self.tokens, offsets=self.offsets, lengths=self.lengths,
                 target_starts=self.target_starts, target_ends=self.target_ends,
                 actions=np.asarray(self.actions))

    @classmethod
    def load(cls, path: Path) -> "Corpus":
        import numpy as np

        data = np.load(path)
        return cls(data["tokens"], data["offsets"], data["lengths"], data["target_starts"],
                   data["target_ends"], [str(a) for a in data["actions"]])


def rows_tensor(corpus: Corpus, rows: Sequence[Sequence[int]], capacity: int, device):
    """Packed rows as (raw tokens [R, capacity], trainable mask, per-record
    (row, start, end, index) target spans)."""
    import torch

    raw = torch.full((len(rows), capacity), PAD_ID, dtype=torch.long)
    trainable = torch.zeros((len(rows), capacity), dtype=torch.bool)
    spans = []
    for r, row in enumerate(rows):
        column = 0
        for index in row:
            offset, length = int(corpus.offsets[index]), int(corpus.lengths[index])
            raw[r, column:column + length] = torch.from_numpy(
                corpus.tokens[offset:offset + length].astype("int64"))
            start = column + int(corpus.target_starts[index])
            end = column + int(corpus.target_ends[index])
            trainable[r, start:end] = True
            spans.append((r, start, end, index))
            column += length
    return raw.to(device, non_blocking=True), trainable.to(device, non_blocking=True), spans


def packed_layout(input_ids):
    """FlashAttention boundaries and per-conversation RoPE positions; same
    rule as nedolm.sft_train._packed_layout (EOS and row starts split)."""
    import torch

    starts = torch.zeros_like(input_ids, dtype=torch.bool)
    starts[:, 0] = True
    starts[:, 1:] = input_ids[:, :-1].eq(EOS_ID)
    offsets = torch.nonzero(starts.reshape(-1), as_tuple=False).flatten().to(torch.int32)
    end = torch.full((1,), input_ids.numel(), device=input_ids.device, dtype=torch.int32)
    cu_seqlens = torch.cat((offsets, end)).contiguous()
    cols = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    last_boundary = torch.cummax(torch.where(starts, cols, torch.zeros_like(cols)), dim=1).values
    return cu_seqlens, (cols - last_boundary).contiguous()


# ---------------------------------------------------------------- training


def forward(model, raw, trainable, *, capacity: int, logits: bool = False):
    import torch

    inputs = raw[:, :-1]
    labels = raw[:, 1:].masked_fill(~trainable[:, 1:], -100)
    cu_seqlens, position_ids = packed_layout(inputs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if logits:
            _, out = model(inputs, return_logits=True, cu_seqlens=cu_seqlens,
                           max_seqlen=capacity - 1, position_ids=position_ids)
            return labels, out
        loss_sum, _ = model(inputs, labels=labels, return_logits=False, loss_reduction="sum",
                            loss_fp32_logits=False, cu_seqlens=cu_seqlens,
                            max_seqlen=capacity - 1, position_ids=position_ids)
    return loss_sum


def evaluate(model, corpus: Corpus, *, capacity: int, rows_per_micro: int, device) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model.eval()
    rows = pack(corpus.lengths, range(len(corpus)), capacity)
    loss_sum, targets = 0.0, 0
    hits: Counter = Counter()
    totals: Counter = Counter()
    with torch.inference_mode():
        for begin in range(0, len(rows), rows_per_micro):
            raw, trainable, spans = rows_tensor(corpus, rows[begin:begin + rows_per_micro], capacity, device)
            labels, logits = forward(model, raw, trainable, capacity=capacity, logits=True)
            mask = labels.ne(-100)
            loss_sum += float(F.cross_entropy(logits[mask].float(), labels[mask], reduction="sum"))
            targets += int(mask.sum())
            predicted = logits.argmax(-1)
            for r, start, end, index in spans:
                hit = bool(torch.equal(predicted[r, start - 1:end - 1], raw[r, start:end]))
                action = corpus.actions[index]
                hits[action] += hit
                totals[action] += 1
            del raw, trainable, labels, logits, predicted
    model.train()
    return {
        "loss": loss_sum / max(1, targets),
        "target_tokens": targets,
        "exact_match": sum(hits.values()) / max(1, sum(totals.values())),
        "exact_match_by_action": {a: round(hits[a] / totals[a], 4) for a in sorted(totals, key=lambda a: -totals[a])},
        "records_by_action": dict(totals),
    }


def epoch_steps(corpus: Corpus, args, epoch: int) -> list[list[list[int]]]:
    """Optimizer steps of one epoch: lists of micro-batches of packed rows.
    The order depends only on the seed and epoch, so a resumed run repeats it."""
    order = list(range(len(corpus)))
    random.Random(args.seed * 1000 + epoch).shuffle(order)
    rows = pack(corpus.lengths, order, args.seq_len)
    micro = [rows[i:i + args.rows_per_micro] for i in range(0, len(rows), args.rows_per_micro)]
    return [micro[i:i + args.grad_accum] for i in range(0, len(micro), args.grad_accum)]


def log_line(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def train(args) -> None:
    import hashlib

    import torch

    sys.path.insert(0, str(args.nedolm_root))
    sys.path.insert(0, str(args.tokenizer_python))
    from nedolm.config import RunConfig
    from nedolm.model import build_model
    from nedotokenizer import SurfaceTokenizer

    run = args.run_dir
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics = run / "metrics.jsonl"

    raw_vocab = args.vocab.read_bytes()
    if hashlib.sha256(raw_vocab).hexdigest() != TOKENIZER_SHA256:
        raise RuntimeError(f"{args.vocab} is not the NDSRF004 vocabulary")
    tokenizer = SurfaceTokenizer(raw_vocab)

    def encode(text: str) -> list[int]:
        ids = list(tokenizer.encode_ids(text.encode("utf-8")))
        if ids and max(ids) >= MESSAGE_START_ID:
            raise RuntimeError("content produced a control id")
        return ids

    if encode("\n") != [13] or encode("asistan\n") != [1986, 751, 13]:
        raise RuntimeError("chat contract mismatch: newline/assistant header ids")

    corpora = {}
    for split in ("train", "validation"):
        cache = run / f"{split}.tokens.npz"
        if cache.exists():
            corpora[split] = Corpus.load(cache)
        else:
            corpora[split] = Corpus.build(args.mix / args.role / f"{split}.jsonl.gz", encode)
            corpora[split].save(cache)
        corpus = corpora[split]
        log_line(metrics, {"event": "data", "split": split, "records": len(corpus),
                           "tokens": int(corpus.lengths.sum()), "target_tokens": corpus.target_tokens,
                           "max_length": int(corpus.lengths.max())})
    train_corpus, val_corpus = corpora["train"], corpora["validation"]

    plans = [epoch_steps(train_corpus, args, epoch) for epoch in range(args.epochs)]
    total_steps = sum(len(p) for p in plans)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * args.warmup_ratio))

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    model = build_model(RunConfig.load(args.model_config).model)
    state = torch.load(args.base, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(state["model"], strict=True)
    del state
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=args.weight_decay, fused=True)

    step, best, best_path = 0, None, None
    last = run / "checkpoints" / "last.pt"
    if last.exists():
        blob = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=True)
        optimizer.load_state_dict(blob["optimizer"])
        step, best, best_path = blob["step"], blob.get("best"), blob.get("best_path")
        del blob
        log_line(metrics, {"event": "resume", "step": step})

    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "total_steps": total_steps, "warmup_steps": warmup,
              "steps_per_epoch": [len(p) for p in plans], "gpu": torch.cuda.get_device_name(0)}
    (run / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    log_line(metrics, {"event": "config", **config})

    def save_last() -> None:
        tmp = last.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "best": best, "best_path": best_path}, tmp)
        os.replace(tmp, last)

    def run_eval(epoch: int, tag: str) -> None:
        nonlocal best, best_path
        result = evaluate(model, val_corpus, capacity=args.seq_len, rows_per_micro=args.rows_per_micro,
                          device=device)
        log_line(metrics, {"event": "eval", "step": step, "epoch": epoch, "tag": tag, **result})
        if tag != "epoch":
            return
        path = run / "checkpoints" / f"epoch_{epoch:02d}_step_{step:06d}.pt"
        torch.save({"model": model.state_dict(), "step": step, "epoch": epoch, "base": str(args.base),
                    "mix": str(args.mix), "role": args.role, "validation": result}, path)
        score = result["exact_match"] if args.select_by == "exact_match" else -result["loss"]
        if best is None or score > best:
            best, best_path = score, str(path)
            (run / "checkpoints" / "best.txt").write_text(path.name + "\n")

    position = 0  # optimizer steps walked through the plans, taken or skipped on resume
    start_time, last_save = time.time(), time.time()
    tokens_seen = 0
    for epoch, plan in enumerate(plans, start=1):
        trained = False
        for micro_batches in plan:
            if step >= total_steps:
                break
            position += 1
            if position <= step:  # resumed: this step was taken before
                continue
            trained = True
            lr = lr_at(step, total_steps, args.lr, warmup, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loaded = [rows_tensor(train_corpus, rows, args.seq_len, device) for rows in micro_batches]
            targets = sum(int(trainable[:, 1:].sum()) for _, trainable, _ in loaded)
            loss_total = 0.0
            for raw, trainable, _ in loaded:
                loss = forward(model, raw, trainable, capacity=args.seq_len) / targets
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}")
                loss.backward()
                loss_total += float(loss)
                tokens_seen += raw.numel()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            step += 1
            if step == 1 or step % args.log_interval == 0:
                elapsed = time.time() - start_time
                log_line(metrics, {"event": "train", "step": step, "epoch": epoch, "loss": round(loss_total, 5),
                                   "lr": lr, "grad_norm": round(grad_norm, 4), "target_tokens": targets,
                                   "tokens_per_s": round(tokens_seen / max(elapsed, 1e-6)),
                                   "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
            if args.eval_interval and step % args.eval_interval == 0:
                run_eval(epoch, "interval")
            if time.time() - last_save > args.save_minutes * 60:
                save_last()
                last_save = time.time()
        if position == sum(len(p) for p in plans[:epoch]):
            if trained:  # an epoch finished before a resume was already evaluated
                run_eval(epoch, "epoch")
                save_last()
            if step >= total_steps:
                break
        else:  # --max-steps stopped mid-epoch (smoke run): report, keep no checkpoint
            if trained:
                run_eval(epoch, "final")
            break
    summary = {"event": "done", "step": step, "best_checkpoint": best_path, "best_score": best,
               "select_by": args.select_by, "hours": round((time.time() - start_time) / 3600, 3)}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    log_line(metrics, summary)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--role", required=True, choices=sorted(ROLE_DEFAULTS))
    parser.add_argument("--mix", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    nedolm = Path("/arf/scratch/egitimg16/nedolm_0p8b")
    parser.add_argument("--base", type=Path, default=nedolm / (
        "runs/nedolm-0p8b-continuation-dpo-v1-step1436-2xh200-prod-20260928/checkpoints/step_00007895.pt"))
    parser.add_argument("--model-config", type=Path,
                        default=nedolm / "sft_im_special_v1/nedolm_base_B_step10397_32k2.yaml")
    parser.add_argument("--nedolm-root", type=Path, default=nedolm)
    parser.add_argument("--tokenizer-python", type=Path, default=Path("/arf/scratch/egitimg16/NedoTokenizer/python"))
    parser.add_argument("--vocab", type=Path,
                        default=Path("/arf/scratch/egitimg16/NedoTokenizer/assets/surface-vocab.bin"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--select-by", choices=("exact_match", "loss"))
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    # One packed row holds the longest record (mix: <= 8192 tokens) plus one shifted token.
    parser.add_argument("--seq-len", type=int, default=8193)
    parser.add_argument("--rows-per-micro", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="smoke runs")
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-minutes", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args(argv)
    for key, value in ROLE_DEFAULTS[args.role].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    train(parse_args())
