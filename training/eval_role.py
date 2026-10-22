"""Generate with a role checkpoint on a mix split and score the outputs.

    python training/eval_role.py --role yurutme --checkpoint <run>/checkpoints/epoch_02_step_002601.pt \\
        --mix <mix/v2> --out <run>/eval_test

Greedy decoding, as the runtime does for Yasama and Yurutme. Writes
``generations.jsonl`` (request, reference, output) and ``report.json``
(training/eval_metrics.py). ``--chat`` also answers a fixed set of general
Turkish requests in Yargi's no-observation format, to compare a Yargi
checkpoint with the DPO base it started from.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.eval_metrics import SCORERS  # noqa: E402
from training.train_role import (  # noqa: E402
    EOS_ID,
    MESSAGE_END_ID,
    MESSAGE_START_ID,
    TOKENIZER_SHA256,
    encode_record,
)

STOP_IDS = {MESSAGE_END_ID, EOS_ID, MESSAGE_START_ID}
DEFAULTS = {  # (records per action, max new tokens)
    "yasama": (0, 16),      # every test record; an action name is a few tokens
    "yurutme": (40, 3072),   # FILE_CREATE contents reach ~3k tokens
    "yargi": (30, 512),
}
CHAT_PROMPTS = (
    "Merhaba, nasılsın?",
    "Fotosentez nedir? İki kısa cümleyle açıkla.",
    "İstanbul hangi iki kıta üzerinde yer alır?",
    "Bir arkadaşım üzgün, ona nasıl destek olabilirim? Üç kısa öneri ver.",
    "Kahve mi çay mı daha sağlıklı?",
    "Yazılım mühendisliğine yeni başlayan birine hangi dili önerirsin, neden?",
    "Dünya neden gece ve gündüz yaşar? Kısaca açıkla.",
    "Bana kısa bir doğum günü mesajı yazar mısın? Annem için.",
    "Stresle başa çıkmak için ne yapabilirim?",
    "Osmanlı İmparatorluğu hangi yıl kuruldu?",
    "Bir CV'de nelere dikkat etmeliyim?",
    "\"Kitap okumayı seviyorum çünkü yeni şeyler öğreniyorum.\" cümlesini daha kısa yaz.",
)


def sample(records: list[dict[str, Any]], per_action: int) -> list[dict[str, Any]]:
    if not per_action:
        return records
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["action"]].append(record)
    chosen = []
    for rows in groups.values():
        rows.sort(key=lambda r: hashlib.md5(f"eval:{r['id']}".encode()).hexdigest())
        chosen += rows[:per_action]
    return chosen


def generate(model, prompts: list[list[int]], max_new: int, budget: int, device) -> list[tuple[list[int], bool]]:
    """Greedy continuation of every prompt; packed recomputation per step (the
    model has no KV cache). Returns (tokens without the stop id, stopped)."""
    import torch
    import torch.nn.functional as F

    weight = model.embed_tokens.weight if model.lm_head is None else model.lm_head.weight
    seqs = [list(p) for p in prompts]
    outs: list[list[int]] = [[] for _ in prompts]
    stopped = [False] * len(prompts)
    with torch.inference_mode():
        for _ in range(max_new):
            active = [i for i in range(len(seqs)) if not stopped[i]]
            if not active:
                break
            chunks, current, used = [], [], 0
            for i in active:
                if current and used + len(seqs[i]) > budget:
                    chunks.append(current)
                    current, used = [], 0
                current.append(i)
                used += len(seqs[i])
            chunks.append(current)
            for chunk in chunks:
                lens = [len(seqs[i]) for i in chunk]
                flat = torch.tensor([t for i in chunk for t in seqs[i]], device=device).unsqueeze(0)
                pos = torch.cat([torch.arange(n, device=device) for n in lens]).unsqueeze(0)
                cums = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device=device, dtype=torch.int32)
                ends = cums[1:].long() - 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, hidden = model(flat, return_hidden=True, position_ids=pos, cu_seqlens=cums,
                                      max_seqlen=max(lens))
                    next_ids = F.linear(hidden[0, ends], weight).float().argmax(-1).tolist()
                for i, token in zip(chunk, next_ids):
                    if token in STOP_IDS:
                        stopped[i] = True
                    else:
                        outs[i].append(token)
                        seqs[i].append(token)
    return list(zip(outs, stopped))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--role", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mix", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-action", type=int, help="records per action label; 0 = all")
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--budget", type=int, default=65536, help="tokens per forward pass")
    parser.add_argument("--chat", action="store_true", help="also answer CHAT_PROMPTS as Yargi")
    nedolm = Path("/arf/scratch/egitimg16/nedolm_0p8b")
    parser.add_argument("--model-config", type=Path, default=nedolm / "sft_im_special_v1/nedolm_base_B_step10397_32k2.yaml")
    parser.add_argument("--nedolm-root", type=Path, default=nedolm)
    parser.add_argument("--tokenizer-python", type=Path, default=Path("/arf/scratch/egitimg16/NedoTokenizer/python"))
    parser.add_argument("--vocab", type=Path, default=Path("/arf/scratch/egitimg16/NedoTokenizer/assets/surface-vocab.bin"))
    args = parser.parse_args()
    per_action = DEFAULTS[args.role][0] if args.per_action is None else args.per_action
    max_new = args.max_new or DEFAULTS[args.role][1]

    import torch

    sys.path.insert(0, str(args.nedolm_root))
    sys.path.insert(0, str(args.tokenizer_python))
    from nedolm.config import RunConfig
    from nedolm.model import build_model
    from nedotokenizer import SurfaceTokenizer

    raw_vocab = args.vocab.read_bytes()
    if hashlib.sha256(raw_vocab).hexdigest() != TOKENIZER_SHA256:
        raise RuntimeError("wrong tokenizer")
    tokenizer = SurfaceTokenizer(raw_vocab)

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode_ids(text.encode("utf-8")))

    def decode(ids: list[int]) -> str:
        text = tokenizer.decode_ids(ids)
        return (text.decode("utf-8", errors="replace") if isinstance(text, (bytes, bytearray)) else str(text)).strip()

    device = torch.device("cuda")
    model = build_model(RunConfig.load(args.model_config).model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(state["model"], strict=True)
    del state
    model.to(device).eval()
    args.out.mkdir(parents=True, exist_ok=True)

    with gzip.open(args.mix / args.role / f"{args.split}.jsonl.gz", "rt", encoding="utf-8") as handle:
        records = sample([json.loads(line) for line in handle], per_action)
    prompts = []
    for record in records:
        ids, start, _ = encode_record(encode, record["messages"])
        prompts.append(ids[:start])
    started = time.time()
    results = generate(model, prompts, max_new, args.budget, device)
    rows = []
    with (args.out / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for record, (tokens, stopped) in zip(records, results):
            row = {"id": record["id"], "action": record["action"], "source": record["source"],
                   "request": record["messages"][-2]["content"], "target": record["messages"][-1]["content"],
                   "output": decode(tokens), "truncated": not stopped}
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"role": args.role, "checkpoint": str(args.checkpoint), "split": args.split,
              "records": len(rows), "truncated": sum(r["truncated"] for r in rows),
              "seconds": round(time.time() - started, 1), "scores": SCORERS[args.role](rows)}

    if args.chat:
        from yargi.yargi import request_messages, system_message

        chat_prompts = []
        for prompt in CHAT_PROMPTS:
            messages = [system_message(), *request_messages(user_prompt=prompt, conversation_history=[], observations=[]),
                        {"role": "assistant", "content": "-"}]
            ids, start, _ = encode_record(encode, messages)
            chat_prompts.append(ids[:start])
        answers = generate(model, chat_prompts, 400, args.budget, device)
        chat = [{"prompt": p, "answer": decode(t), "truncated": not s} for p, (t, s) in zip(CHAT_PROMPTS, answers)]
        (args.out / "chat.json").write_text(json.dumps(chat, ensure_ascii=False, indent=2) + "\n")
        report["chat_answers"] = len(chat)

    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False)[:4000], flush=True)


if __name__ == "__main__":
    main()
