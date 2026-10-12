"""KL-matched random-LoRA control for a group of trained adapters (the sprint's `noise` reference point,
kaushik/noise/README.md); one copy for gepa_lora / lora_arith / len_teacher.

    run.py elicit/gepa_lora noise noise.group=G "noise.adapters=[/checkpoints/...,...]" [--dry-run]
    run.py sweep elicit/lora_arith/sweeps/noise.yaml --dry-run

1. KL(base || adapter) per token for every trained seed on the base model's own T=1 samples (`noise.samples`:
   200 unconstrained val questions, cached by kaushik/noise and reused read-only); the base is rescored through
   the same engine path (KL(base||base) must be ~0).
2. target = median KL over seeds; the seed closest to the median lends its shapes / per-module norm profile.
3. `noise.seeds` random adapters (family `noise.family`) with the scale calibrated until KL is within
   `noise.tol` of the target, built on the workstream's random-lora CPU app
   (cotcontrol.training.random_lora.build_adapter) at <noise.out_root>/<group>/noise<k>.
-> runs/_noise/<group>/{kl.json, noise_adapters.json, noise.log}. Evaluate the adapters afterwards with a
candidate sweep (candidate.lora: <noise.out_root>/<group>/noise<k>, test split).
"""

# ruff: noqa: DTZ005 — naive local timestamps, like the rest of the repo
from __future__ import annotations

import argparse
import asyncio
import statistics
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from kaushik.core.io import ROOT, atomic_write_json, read_json, rel
from kaushik.core.scheduler import Job, run_scheduler
from kaushik.elicit.common import (
    blocks_of,
    check_modal,
    gen_config_of,
    runs_dir,
    setup_modal,
)


def out_dir(cfg: dict) -> Path:
    """runs/_noise/<group> (next to the run dirs, whatever run_name the sweep gives the job)."""
    return runs_dir(cfg) / "_noise" / cfg["noise"]["group"]


async def run(cfg: dict) -> dict:
    from cotcontrol.training import random_lora
    from cotcontrol.training.random_lora import (
        calibrate_scale,
        kl_from_logprobs,
        score_samples,
    )

    nz = cfg["noise"]
    group, adapters, seeds, tol = nz["group"], list(nz["adapters"]), [int(s) for s in nz["seeds"]], float(nz["tol"])
    base_model = cfg["base_model"]
    od = out_dir(cfg)
    od.mkdir(parents=True, exist_ok=True)
    samples = read_json(ROOT / nz["samples"])
    blk = blocks_of(cfg, ["modes"])["modes"]
    assert samples["base_model"] == base_model and \
        samples["generate_config"]["chat_template_kwargs"] == blk["chat_template_kwargs"], "samples/protocol mismatch"
    gen = replace(gen_config_of(cfg, blk), return_logprobs=True, **{k: v for k, v in nz["scoring"].items()})
    log = open(od / "noise.log", "a")  # noqa: SIM115, ASYNC230 - appended to across awaits for the whole calibration
    sem = asyncio.Semaphore(int(nz["gpus"]))

    def say(s: str) -> None:
        print(s, flush=True)
        log.write(f"[{datetime.now().isoformat()}] {s}\n")
        log.flush()

    async def kl_of(path: str | None, base_lps):
        async with sem:
            lps = await score_samples(samples, base_model, gen, path)
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
                      od / "kl.json", indent=1)
    root, calib = f"{nz['out_root']}/{group}", f"{nz['out_root']}/{group}/_calib"
    build = random_lora.remote_fn("build_adapter")
    family = nz["family"]

    def kl_at_for(seed: int):
        async def kl_at(scale: float) -> float:
            path = f"{calib}/s{seed}_x{scale:.6g}"
            await build.remote.aio(ref, path, scale, seed, family)
            k = (await kl_of(path, base_lps))["kl_per_token"]
            say(f"  seed{seed} scale {scale:.4g}: KL {k:.5f} (ratio {k / target:.3f})")
            return k
        return kl_at

    async def finish(k: int, seed: int, scale0: float) -> dict:
        scale, hist = await calibrate_scale(kl_at_for(seed), target, scale0=scale0, tol=tol)
        final = f"{root}/noise{k}"
        b = await build.remote.aio(ref, final, scale, seed, family)
        res = await kl_of(final, base_lps)
        say(f"noise{k} (seed {seed}): scale {scale:.4g} KL {res['kl_per_token']:.5f} after {len(hist)} evals -> {final}")
        return {"k": k, "seed": seed, "scale": scale, "path": final, "kl": res, "calibration": hist, "build": b}

    first = await finish(0, seeds[0], 1.0)
    rest = await asyncio.gather(*[finish(k, s, first["scale"]) for k, s in enumerate(seeds[1:], 1)])
    await random_lora.remote_fn("remove_path").remote.aio(calib)
    rec = {"group": group, "base_model": base_model, "target_kl": target,
           "trained_kl": {a: k["kl_per_token"] for a, k in kls.items()}, "reference": ref, "family": family,
           "tolerance": tol, "engine": asdict(gen), "samples": nz["samples"], "adapters": [first, *rest],
           "timestamp": datetime.now().astimezone().isoformat()}
    atomic_write_json(rec, od / "noise_adapters.json", indent=1)
    say(f"-> {rel(od / 'noise_adapters.json')}")
    return rec


def jobs(cfg: dict) -> list[Job]:
    setup_modal(cfg)
    nz = cfg["noise"]
    if not nz.get("group") or not nz.get("adapters"):
        raise ValueError("noise.group and noise.adapters must be set")
    od = out_dir(cfg)
    return [Job(name=f"noise:{nz['group']}", weight=int(nz["gpus"]), run=lambda: run(cfg),
                done=lambda: (od / "noise_adapters.json").exists(), priority=(2, nz["group"]), group=str(od))]


def main(cfg: dict, argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="noise", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    nz = cfg["noise"]
    if a.dry_run:
        print(f"noise group {nz['group']}: adapters {nz['adapters']}, seeds {nz['seeds']}, tol {nz['tol']}, "
              f"family {nz['family']} -> {nz['out_root']}/{nz['group']}")
    else:
        check_modal(cfg)
    run_scheduler(jobs(cfg), runs_dir(cfg), cap=a.cap, name="noise_scheduler", dry_run=a.dry_run)
