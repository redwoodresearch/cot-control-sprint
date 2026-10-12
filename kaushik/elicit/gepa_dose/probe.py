"""Does vLLM honour the scaled lora_alpha? Teacher-forced log-prob of the TEACHER's first reasoning tokens under
base and the `probe.doses` adapters (`probe.seed`), on `probe.n` held-out TRAIN prompts (B2's prompts.jsonl,
holdout rows). If the dose is applied, logp(teacher tokens) should move monotonically with dose (x0.5 between
base and x1; x2/x3 beyond x1, possibly overshooting). Writes runs/_probe/dose.json.

    run.py elicit/gepa_dose probe [probe.n=8] [probe.max_tokens=24]
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from kaushik.core.io import ROOT, atomic_write_json, read_jsonl
from kaushik.elicit.common import (
    blocks_of,
    check_modal,
    gen_config_of,
    runs_dir,
    setup_modal,
)
from kaushik.elicit.common.distill import teacher_prompt


def adapter_path(cfg: dict, seed: int, dose: float) -> str:
    """<dose.vol_dir>/s<seed>_x<dose>; dose x1 is B2's original adapter."""
    d = cfg["dose"]
    return d["b2_adapter"].format(seed=seed) if dose == 1 else f"{d['vol_dir']}/s{seed}_x{dose:g}"


async def run(cfg: dict) -> dict:
    from cotcontrol.inference.modal_vllm import generate_async, score_async

    p = cfg["probe"]
    base_model = cfg["base_model"]
    n = int(p["n"])
    rows = [r for r in read_jsonl(ROOT / cfg["distill"]["data"]["prompts"]) if r["holdout"]]
    rows = rows[:: max(1, len(rows) // n)][:n]
    msgs_s = [[{"role": "user", "content": r["user"]}] for r in rows]
    msgs_t = [[{"role": "system", "content": teacher_prompt(cfg)}] + m for m in msgs_s]
    gen = replace(gen_config_of(cfg, blocks_of(cfg, ["modes"])["modes"]), temperature=0.0,
                  max_tokens=int(p["max_tokens"]), cache=False, return_logprobs=True)
    base, teacher = await asyncio.gather(generate_async(msgs_s, base_model, gen, None),
                                         generate_async(msgs_t, base_model, gen, None))
    seqs = [list(b["prompt_token_ids"]) + list(t["token_ids"][0]) for b, t in zip(base, teacher)]
    plens = [len(b["prompt_token_ids"]) for b in base]
    doses = list(p["doses"])
    paths = [None if d == 0 else adapter_path(cfg, int(p["seed"]), d) for d in doses]
    lps = await asyncio.gather(*[score_async(seqs, plens, base_model, gen, path) for path in paths])
    out = {"n": len(rows), "doses": doses, "paths": paths,
           "teacher_first_tokens": [t["metadata"][0]["raw_response"] for t in teacher],
           "logp_teacher_tokens": {str(d): [sum(x) for x in lp] for d, lp in zip(doses, lps)},
           "first_tok_logp": {str(d): [x[0] for x in lp] for d, lp in zip(doses, lps)}}
    out["mean_logp"] = {d: sum(v) / len(v) for d, v in out["logp_teacher_tokens"].items()}
    out["mean_first_tok_logp"] = {d: sum(v) / len(v) for d, v in out["first_tok_logp"].items()}
    return out


def main(cfg: dict, argv: list[str]) -> None:
    setup_modal(cfg)
    check_modal(cfg)
    out = asyncio.run(run(cfg))
    atomic_write_json(out, runs_dir(cfg) / "_probe" / "dose.json", indent=1)
    print("dose -> mean logp(teacher first tokens) | mean first-token logp")
    for d in out["doses"]:
        print(f"  x{d:<4g} {out['mean_logp'][str(d)]:8.2f} | {out['mean_first_tok_logp'][str(d)]:7.3f}")
