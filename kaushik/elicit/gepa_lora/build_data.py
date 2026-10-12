"""Build the distillation prompt set: TRAIN-split questions x the 9 training constraint modes (never val/test).

    run.py elicit/gepa_lora build_data

Writes <data.out>/prompts.jsonl: {"qid", "dataset", "mode", "user", "holdout"}. `holdout` marks `data.holdout_frac`
of the questions (by question, seeded) kept out of the gradient to track the distillation loss. drift.jsonl holds
the unconstrained (baseline-mode) prompts of the held-out questions, used to watch KL(base || student) drift on
prompts that carry no Requirement. Rows ship to the Modal trainer as-is; the teacher prompt is added there.
"""

from __future__ import annotations

import hashlib
import random

from kaushik.core.io import ROOT, atomic_write_json, write_jsonl
from kaushik.elicit.common import sha256_text
from kaushik.elicit.common.distill import teacher_prompt


def main(cfg: dict, argv: list[str]) -> None:
    from cotcontrol.eval.data import assign_tasks, load_dataset
    from cotcontrol.eval.prompts import create_user_prompt

    d = cfg["data"]
    out = ROOT / d["out"]
    base_model = cfg["base_model"]
    samples = load_dataset("all", "all", split=d["split"])
    qids = sorted({f"{s['dataset']}:{s['id']}" for s in samples})
    rng = random.Random(int(d["seed"]))
    holdout = set(rng.sample(qids, round(float(d["holdout_frac"]) * len(qids))))

    rows = []
    for s, m in assign_tasks(samples, "all", seed=int(d["assign_seed"])):
        qid = f"{s['dataset']}:{s['id']}"
        rows.append({"qid": qid, "dataset": s["dataset"], "mode": m,
                     "user": create_user_prompt(s, m, base_model), "holdout": qid in holdout})
    drift = [{"qid": f"{s['dataset']}:{s['id']}", "dataset": s["dataset"], "mode": "baseline",
              "user": create_user_prompt(s, "baseline", base_model), "holdout": True}
             for s in samples if f"{s['dataset']}:{s['id']}" in holdout]
    write_jsonl(out / "prompts.jsonl", rows)
    write_jsonl(out / "drift.jsonl", drift)
    n_h = sum(r["holdout"] for r in rows)
    meta = {"n_questions": len(qids), "n_holdout_questions": len(holdout), "n_prompts": len(rows),
            "n_holdout_prompts": n_h, "n_train_prompts": len(rows) - n_h, "n_drift": len(drift),
            "modes": sorted({r["mode"] for r in rows}), "split": d["split"], "seed": int(d["seed"]),
            "gepa_prompt_sha256": sha256_text(teacher_prompt(cfg)),
            "prompts_sha256": hashlib.sha256((out / "prompts.jsonl").read_bytes()).hexdigest()}
    atomic_write_json(meta, out / "meta.json", indent=1)
    print(meta)
