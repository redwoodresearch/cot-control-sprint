"""Sanity probe in the vLLM eval path, before any eval:

    kaushik/run.py elicit/gepa_lora probe --lora /checkpoints/gepa_lora/<run>/checkpoint-200 [--n 8]

On n held-out TRAIN-split prompts: greedy first 24 tokens from (base, no system prompt), (student = base+LoRA,
no system prompt) and (teacher = base + GEPA prompt); then teacher-forced log-prob of the TEACHER's first tokens
under base vs student (same student-side prompt). The adapter "moves toward the teacher" if
logp_student(teacher tokens) > logp_base(teacher tokens) on most prompts. Writes runs/_probe/<ckpt>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys

from gl_common import BASE_MODEL, DATA, GEN_KWARGS, RUNS, atomic_write_json, check_apps, gepa_prompt, read_jsonl

sys.path.insert(0, str(RUNS.parents[3]))
from cotcontrol.inference.modal_vllm import ModalGenerateConfig, generate_async, score_async  # noqa: E402


async def run(lora: str, n: int, max_tokens: int) -> dict:
    rows = [r for r in read_jsonl(DATA / "prompts.jsonl") if r["holdout"]]
    rows = rows[:: max(1, len(rows) // n)][:n]
    msgs_s = [[{"role": "user", "content": r["user"]}] for r in rows]
    msgs_t = [[{"role": "system", "content": gepa_prompt()}] + m for m in msgs_s]
    gen = ModalGenerateConfig(**{**GEN_KWARGS, "temperature": 0.0, "max_tokens": max_tokens, "cache": False,
                                 "return_logprobs": True})
    base, student, teacher = await asyncio.gather(
        generate_async(msgs_s, BASE_MODEL, gen, None), generate_async(msgs_s, BASE_MODEL, gen, lora),
        generate_async(msgs_t, BASE_MODEL, gen, None))
    # Teacher-forced: student-side prompt + the teacher's generated tokens, scored under base and under the LoRA.
    seqs = [list(b["prompt_token_ids"]) + list(t["token_ids"][0]) for b, t in zip(base, teacher)]
    plens = [len(b["prompt_token_ids"]) for b in base]
    lp_base, lp_stud = await asyncio.gather(score_async(seqs, plens, BASE_MODEL, gen, None),
                                            score_async(seqs, plens, BASE_MODEL, gen, lora))
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args(argv)
    check_apps()
    out = asyncio.run(run(args.lora, args.n, args.max_tokens))
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.lora.replace("/checkpoints/gepa_lora/", ""))
    atomic_write_json(out, RUNS / "_probe" / f"{name}.json")
    for it in out["items"]:
        print(f"--- {it['qid']} [{it['mode']}]  logp(teacher toks): base {it['logp_teacher_tokens_under_base']:.2f} "
              f"student {it['logp_teacher_tokens_under_student']:.2f}")
        for k in ("base", "student", "teacher"):
            print(f"  {k:8s} {it[k][:160]!r}")
    print(out["summary"])


if __name__ == "__main__":
    main()
