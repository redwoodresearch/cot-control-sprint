"""KL-matched random-LoRA control for the GEPA LoRA (the sprint's `noise` reference point; kaushik/noise/README.md).

    kaushik/run.py elicit/gepa_lora noise --group G --adapters /checkpoints/gepa_lora/<run_s0>/checkpoint-N,...

1. KL(base || adapter) per token for every trained seed on the base model's own T=1 samples (200 unconstrained val
   questions, 2048 tokens, reasoning_effort medium) — the samples cached by kaushik/noise (read-only reuse); the
   base is rescored here through the same engine path (KL(base||base) must be ~0).
2. target = median KL over seeds; the seed closest to the median lends its shapes / per-module norm profile.
3. 3 random adapters (noise seeds 0/1/2, family frob) with scale calibrated until KL is within 10% of target,
   built on the kaushik/noise CPU app (cotcontrol.training.random_lora.build_adapter) at
   /checkpoints/gepa_lora/noise/<G>/noise{k}. -> runs/_noise/<G>/noise_adapters.json
Evaluate them afterwards with `launch` (--lora /checkpoints/gepa_lora/noise/<G>/noise{k}, test split).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
from dataclasses import asdict, replace
from datetime import datetime

from gl_common import BASE_MODEL, GEN_KWARGS, ROOT, RUNS, VOL_DIR, atomic_write_json, check_apps, load_json

sys.path.insert(0, str(ROOT))
os.environ.setdefault("COTCONTROL_MODAL_RANDOM_LORA_APP", "cot-control-sprint_kaushik-reddy_sub-gepa-lora_random-lora")
from cotcontrol.inference.modal_vllm import ModalGenerateConfig  # noqa: E402
from cotcontrol.training import random_lora  # noqa: E402
from cotcontrol.training.random_lora import calibrate_scale, kl_from_logprobs, score_samples  # noqa: E402

SAMPLES = ROOT / "kaushik/noise/runs/_samples/gptoss20b-82bf9e/samples.json"


async def run(group: str, adapters: list[str], seeds: list[int], tol: float) -> dict:
    out_dir = RUNS / "_noise" / group
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_json(SAMPLES)
    assert samples["base_model"] == BASE_MODEL and samples["generate_config"]["chat_template_kwargs"] == GEN_KWARGS["chat_template_kwargs"]
    gen = replace(ModalGenerateConfig(**GEN_KWARGS), temperature=1.0, max_tokens=2048, seed=0, return_logprobs=True)
    log = open(out_dir / "noise.log", "a")

    def say(s: str) -> None:
        print(s, flush=True)
        log.write(f"[{datetime.now().isoformat()}] {s}\n")
        log.flush()

    from evalrun import _slots
    slots = _slots()

    async def kl_of(path: str | None, base_lps):
        async with slots.slot(1):  # share the workstream GPU cap with the evals
            lps = await score_samples(samples, BASE_MODEL, gen, path)
        return kl_from_logprobs(base_lps, lps) if base_lps is not None else lps

    base_lps = await kl_of(None, None)
    say(f"base rescored: {sum(len(x) for x in base_lps)} tokens; KL(base||base)={kl_from_logprobs(base_lps, base_lps, 10)['kl_per_token']:.6f}")
    kls = dict(zip(adapters, await asyncio.gather(*[kl_of(a, base_lps) for a in adapters])))
    for a, k in kls.items():
        say(f"KL(base||{a}) = {k['kl_per_token']:.5f} ± {k['se_prompt']:.5f} nats/token")
    target = statistics.median(k["kl_per_token"] for k in kls.values())
    ref = min(adapters, key=lambda a: abs(kls[a]["kl_per_token"] - target))
    say(f"target KL (median over {len(adapters)} seeds) = {target:.5f}; reference shapes from {ref}")
    atomic_write_json({"group": group, "adapters": adapters, "kl": kls, "target": target, "reference": ref},
                      out_dir / "kl.json")
    root, calib = f"{VOL_DIR}/noise/{group}", f"{VOL_DIR}/noise/{group}/_calib"
    build = random_lora.remote_fn("build_adapter")

    def kl_at_for(seed: int):
        async def kl_at(scale: float) -> float:
            path = f"{calib}/s{seed}_x{scale:.6g}"
            await build.remote.aio(ref, path, scale, seed, "frob")
            k = (await kl_of(path, base_lps))["kl_per_token"]
            say(f"  seed{seed} scale {scale:.4g}: KL {k:.5f} (ratio {k / target:.3f})")
            return k
        return kl_at

    async def finish(k: int, seed: int, scale0: float) -> dict:
        scale, hist = await calibrate_scale(kl_at_for(seed), target, scale0=scale0, tol=tol)
        final = f"{root}/noise{k}"
        b = await build.remote.aio(ref, final, scale, seed, "frob")
        res = await kl_of(final, base_lps)
        say(f"noise{k} (seed {seed}): scale {scale:.4g} KL {res['kl_per_token']:.5f} after {len(hist)} evals -> {final}")
        return {"k": k, "seed": seed, "scale": scale, "path": final, "kl": res, "calibration": hist, "build": b}

    first = await finish(0, seeds[0], 1.0)
    rest = await asyncio.gather(*[finish(k, s, first["scale"]) for k, s in enumerate(seeds[1:], 1)])
    await random_lora.remote_fn("remove_path").remote.aio(calib)
    rec = {"group": group, "base_model": BASE_MODEL, "target_kl": target, "trained_kl": {a: k["kl_per_token"] for a, k in kls.items()},
           "reference": ref, "family": "frob", "tolerance": tol, "engine": asdict(gen), "samples": str(SAMPLES.relative_to(ROOT)),
           "adapters": [first, *rest], "timestamp": datetime.now().astimezone().isoformat()}
    atomic_write_json(rec, out_dir / "noise_adapters.json")
    say(f"-> {out_dir / 'noise_adapters.json'}")
    return rec


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--adapters", required=True, help="comma-separated /checkpoints/... adapter dirs (the trained seeds)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tol", type=float, default=0.1)
    args = ap.parse_args(argv)
    check_apps()
    asyncio.run(run(args.group, args.adapters.split(","), [int(s) for s in args.seeds.split(",")], args.tol))


if __name__ == "__main__":
    main()
