"""Sanity probe in the vLLM eval path, before any eval:

    run.py elicit/gepa_lora probe probe.lora=/checkpoints/gepa_lora/<run>/checkpoint-200 [probe.n=8]

On `probe.n` held-out TRAIN-split prompts: greedy first `probe.max_tokens` tokens from (base, no system prompt),
(student = base+LoRA, no system prompt) and (teacher = base + the distillation teacher prompt); then
teacher-forced log-prob of the TEACHER's first tokens under base vs student (same student-side prompt). The
adapter "moves toward the teacher" if logp_student(teacher tokens) > logp_base(teacher tokens) on most prompts.
Writes runs/_probe/<ckpt>.json.
"""

from __future__ import annotations

import asyncio
import re

from kaushik.core.io import ROOT, atomic_write_json, read_jsonl
from kaushik.elicit.common import (
    blocks_of,
    check_modal,
    gen_config_of,
    runs_dir,
    setup_modal,
)
from kaushik.elicit.common.distill import teacher_prompt


async def run(cfg: dict, lora: str, n: int, max_tokens: int) -> dict:
    from dataclasses import replace

    from cotcontrol.inference.modal_vllm import generate_async, score_async

    base_model = cfg["base_model"]
    rows = [r for r in read_jsonl(ROOT / cfg["distill"]["data"]["prompts"]) if r["holdout"]]
    rows = rows[:: max(1, len(rows) // n)][:n]
    msgs_s = [[{"role": "user", "content": r["user"]}] for r in rows]
    msgs_t = [[{"role": "system", "content": teacher_prompt(cfg)}] + m for m in msgs_s]
    gen = replace(gen_config_of(cfg, blocks_of(cfg, ["modes"])["modes"]), temperature=0.0, max_tokens=max_tokens,
                  cache=False, return_logprobs=True)
    base, student, teacher = await asyncio.gather(
        generate_async(msgs_s, base_model, gen, None), generate_async(msgs_s, base_model, gen, lora),
        generate_async(msgs_t, base_model, gen, None))
    # Teacher-forced: student-side prompt + the teacher's generated tokens, scored under base and under the LoRA.
    seqs = [list(b["prompt_token_ids"]) + list(t["token_ids"][0]) for b, t in zip(base, teacher)]
    plens = [len(b["prompt_token_ids"]) for b in base]
    lp_base, lp_stud = await asyncio.gather(score_async(seqs, plens, base_model, gen, None),
                                            score_async(seqs, plens, base_model, gen, lora))
    out = {"lora": lora, "n": len(rows), "items": []}
    for r, b, s, t, lb, ls in zip(rows, base, student, teacher, lp_base, lp_stud):
        out["items"].append({"qid": r["qid"], "mode": r["mode"],
                             "base": b["metadata"][0]["raw_response"], "student": s["metadata"][0]["raw_response"],
                             "teacher": t["metadata"][0]["raw_response"],
                             "logp_teacher_tokens_under_base": sum(lb), "logp_teacher_tokens_under_student": sum(ls),
                             "first_tok_lp_base": lb[0], "first_tok_lp_student": ls[0]})
    wins = sum(i["logp_teacher_tokens_under_student"] > i["logp_teacher_tokens_under_base"] for i in out["items"])
    out["summary"] = {"student_closer_to_teacher": f"{wins}/{len(rows)}",
                      "mean_logp_gain": sum(i["logp_teacher_tokens_under_student"] - i["logp_teacher_tokens_under_base"]
                                            for i in out["items"]) / len(rows),
                      "mean_first_tok_gain": sum(i["first_tok_lp_student"] - i["first_tok_lp_base"]
                                                 for i in out["items"]) / len(rows)}
    return out


def main(cfg: dict, argv: list[str]) -> None:
    p = cfg["probe"]
    if not p.get("lora"):
        raise SystemExit("set probe.lora=<adapter path>")
    setup_modal(cfg)
    check_modal(cfg)
    out = asyncio.run(run(cfg, p["lora"], int(p["n"]), int(p["max_tokens"])))
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", p["lora"].replace(f"{cfg['distill']['out_root']}/", ""))
    atomic_write_json(out, runs_dir(cfg) / "_probe" / f"{name}.json", indent=1)
    for it in out["items"]:
        print(f"--- {it['qid']} [{it['mode']}]  logp(teacher toks): base {it['logp_teacher_tokens_under_base']:.2f} "
              f"student {it['logp_teacher_tokens_under_student']:.2f}")
        for k in ("base", "student", "teacher"):
            print(f"  {k:8s} {it[k][:160]!r}")
    print(out["summary"])
