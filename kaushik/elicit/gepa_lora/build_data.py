"""Build the distillation prompt set: TRAIN-split questions x the 9 training constraint modes (never val/test).

    kaushik/run.py elicit/gepa_lora build_data

Writes data/prompts.jsonl: {"qid", "dataset", "mode", "user", "holdout"}. `holdout` marks ~10% of questions
(by question, seeded) kept out of the gradient to track the distillation loss. data/drift.jsonl holds the
unconstrained (baseline-mode) prompts of the held-out questions, used to watch KL(base || student) drift on
prompts that carry no Requirement. Rows ship to the Modal trainer as-is; the GEPA prompt is added there.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys

from gl_common import BASE_MODEL, DATA, ROOT, atomic_write_json, gepa_prompt_sha256, write_jsonl

sys.path.insert(0, str(ROOT))
from cotcontrol.eval.data import assign_tasks, load_dataset  # noqa: E402
from cotcontrol.eval.prompts import create_user_prompt  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    samples = load_dataset("all", "all", split="train")
    qids = sorted({f"{s['dataset']}:{s['id']}" for s in samples})
    rng = random.Random(args.seed)
    holdout = set(rng.sample(qids, int(round(args.holdout_frac * len(qids)))))

    rows = []
    for s, m in assign_tasks(samples, "all", seed=0):
        qid = f"{s['dataset']}:{s['id']}"
        rows.append({"qid": qid, "dataset": s["dataset"], "mode": m,
                     "user": create_user_prompt(s, m, BASE_MODEL), "holdout": qid in holdout})
    drift = [{"qid": f"{s['dataset']}:{s['id']}", "dataset": s["dataset"], "mode": "baseline",
              "user": create_user_prompt(s, "baseline", BASE_MODEL), "holdout": True}
             for s in samples if f"{s['dataset']}:{s['id']}" in holdout]
    write_jsonl(rows, DATA / "prompts.jsonl")
    write_jsonl(drift, DATA / "drift.jsonl")
    n_h = sum(r["holdout"] for r in rows)
    meta = {"n_questions": len(qids), "n_holdout_questions": len(holdout), "n_prompts": len(rows),
            "n_holdout_prompts": n_h, "n_train_prompts": len(rows) - n_h, "n_drift": len(drift),
            "modes": sorted({r["mode"] for r in rows}), "split": "train", "seed": args.seed,
            "gepa_prompt_sha256": gepa_prompt_sha256(),
            "prompts_sha256": hashlib.sha256((DATA / "prompts.jsonl").read_bytes()).hexdigest()}
    atomic_write_json(meta, DATA / "meta.json")
    print(meta)


if __name__ == "__main__":
    main()
