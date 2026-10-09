"""Does vLLM honour the scaled lora_alpha? Teacher-forced log-prob of the TEACHER's first reasoning tokens under
base, x0.5, x1, x1.5, x2, x3 (seed-0 adapter), on n held-out TRAIN prompts (B2's data/prompts.jsonl, holdout rows).
If the dose is applied, logp(teacher tokens) should move monotonically with dose (x0.5 between base and x1; x2/x3
beyond x1, possibly overshooting). Writes runs/_probe/dose.json.

    kaushik/run.py elicit/gepa_dose probe [--n 8] [--max-tokens 24]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from dose_common import (BASE_MODEL, DOSES, GEN_KWARGS, GL, ROOT, RUNS, adapter_path, atomic_write_json, check_apps,
                         gepa_prompt, read_jsonl)

sys.path.insert(0, str(ROOT))
from cotcontrol.inference.modal_vllm import ModalGenerateConfig, generate_async, score_async  # noqa: E402


async def run(n: int, max_tokens: int) -> dict:
    rows = [r for r in read_jsonl(GL / "data" / "prompts.jsonl") if r["holdout"]]
    rows = rows[:: max(1, len(rows) // n)][:n]
    msgs_s = [[{"role": "user", "content": r["user"]}] for r in rows]
    msgs_t = [[{"role": "system", "content": gepa_prompt()}] + m for m in msgs_s]
    gen = ModalGenerateConfig(**{**GEN_KWARGS, "temperature": 0.0, "max_tokens": max_tokens, "cache": False,
                                 "return_logprobs": True})
    base, teacher = await asyncio.gather(generate_async(msgs_s, BASE_MODEL, gen, None),
                                         generate_async(msgs_t, BASE_MODEL, gen, None))
    seqs = [list(b["prompt_token_ids"]) + list(t["token_ids"][0]) for b, t in zip(base, teacher)]
    plens = [len(b["prompt_token_ids"]) for b in base]
    doses = [0, 0.5, 1, *[d for d in DOSES if d > 1]]
    paths = [None if d == 0 else adapter_path(0, d) for d in doses]
    lps = await asyncio.gather(*[score_async(seqs, plens, BASE_MODEL, gen, p) for p in paths])
    out = {"n": len(rows), "doses": doses, "paths": paths, "teacher_first_tokens": [t["metadata"][0]["raw_response"] for t in teacher],
           "logp_teacher_tokens": {str(d): [sum(x) for x in lp] for d, lp in zip(doses, lps)},
           "first_tok_logp": {str(d): [x[0] for x in lp] for d, lp in zip(doses, lps)}}
    out["mean_logp"] = {d: sum(v) / len(v) for d, v in out["logp_teacher_tokens"].items()}
    out["mean_first_tok_logp"] = {d: sum(v) / len(v) for d, v in out["first_tok_logp"].items()}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args(argv)
    check_apps()
    out = asyncio.run(run(args.n, args.max_tokens))
    atomic_write_json(out, RUNS / "_probe" / "dose.json")
    print("dose -> mean logp(teacher first tokens) | mean first-token logp")
    for d in out["doses"]:
        print(f"  x{d:<4g} {out['mean_logp'][str(d)]:8.2f} | {out['mean_first_tok_logp'][str(d)]:7.3f}")


if __name__ == "__main__":
    main()
