"""Run a role's mix test records through a .mercan export with libmercan, as
the runtime does, and compare with the PyTorch checkpoint's generations from
training/eval_role.py: quantization, chat formatting and sampler parity.

    python training/runtime_parity.py --role yasama --model Mercan-0.8B-SoP-Yasama-mixv3-Q4_K_M.mercan \\
        --mix-test yasama/test.jsonl.gz --generations eval_test/generations.jsonl --out parity/yasama

Records are those of ``--generations`` (``--per-action`` caps them); the
generation config is the role's trained runtime default unless overridden.
Writes ``generations.jsonl`` and ``report.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.libmercan_backend import LibMercanBackend  # noqa: E402
from training.eval_metrics import SCORERS  # noqa: E402
from training.eval_role import sample  # noqa: E402
from yargi import yargi  # noqa: E402
from yasama import yasama  # noqa: E402
from yurutme import yurutme  # noqa: E402

ROLE_MODULES = {"yasama": yasama, "yurutme": yurutme, "yargi": yargi}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--role", required=True, choices=sorted(ROLE_MODULES))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mix-test", required=True, type=Path)
    parser.add_argument("--generations", required=True, type=Path, help="eval_role.py generations.jsonl")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-action", type=int, default=0, help="records per action label; 0 = all")
    parser.add_argument("--repeat-penalty", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-ctx", type=int, default=12288)
    args = parser.parse_args()

    config = ROLE_MODULES[args.role].default_generation_config(trained=True)
    overrides = {"repeat_penalty": args.repeat_penalty, "max_tokens": args.max_tokens}
    config = dataclasses.replace(config, **{k: v for k, v in overrides.items() if v is not None})
    reference = {}
    with args.generations.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            reference[row["id"]] = row
    with gzip.open(args.mix_test, "rt", encoding="utf-8") as handle:
        records = [r for r in map(json.loads, handle) if r["id"] in reference]
    records = sample(records, args.per_action)

    args.out.mkdir(parents=True, exist_ok=True)
    rows, torch_rows = [], []
    started = time.time()
    with LibMercanBackend(args.model, device=args.device, n_ctx=args.n_ctx) as backend, \
            (args.out / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, 1):
            output = backend.generate(record["messages"][:-1], config=config).strip()
            torch_row = reference[record["id"]]
            row = {**torch_row, "output": output, "torch_output": torch_row["output"],
                   "same_as_torch": output == torch_row["output"].strip()}
            rows.append(row)
            torch_rows.append(torch_row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 50 == 0:
                print(f"{index}/{len(records)} same={sum(r['same_as_torch'] for r in rows) / index:.3f} "
                      f"{time.time() - started:.0f}s", flush=True)
    report = {
        "role": args.role, "model": str(args.model), "records": len(rows),
        "generation_config": dataclasses.asdict(config), "seconds": round(time.time() - started, 1),
        "same_as_torch": round(sum(r["same_as_torch"] for r in rows) / max(1, len(rows)), 4),
        "scores": SCORERS[args.role](rows), "torch_scores": SCORERS[args.role](torch_rows),
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False)[:3000], flush=True)


if __name__ == "__main__":
    main()
