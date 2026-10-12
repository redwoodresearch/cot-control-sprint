"""Calibrated noise adapters of one group (a point of sweeps/groups.yaml).

    .venv/bin/python kaushik/run.py sweep noise/sweeps/groups.yaml --dry-run
    .venv/bin/python kaushik/run.py noise make [--group G] [--dry-run] [--force]

Jobs per group (after every seed's KL record exists and the group's seeds are complete, or
`groups.wait_hours` have passed since `watch` first saw it):
  manifest:<G>         runs/<G>/manifest.json (group record, dataset / source-config hashes, registry snapshot)
  make:<G>/noise<k>    target = median KL of the trained seeds; the seed closest to the median lends its
                       shapes and per-module norms; `scale` is calibrated (bracket, log-log interpolation,
                       <= noise.max_evals build+score rounds) until KL is within noise.tolerance of the
                       target -> noise.out_root/noise<k> + runs/<G>/make/noise<k>.json. noise0 starts at
                       noise.scale0, later seeds at noise0's scale, so they run in parallel after it.
  make:<G>/finish      runs/<G>/noise_adapters.json (target, reference, calibration history per adapter);
                       removes the calibration scratch on the volume.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from kaushik.core.config import defaults, train_eval_config
from kaushik.core.io import (
    KAUSHIK,
    ROOT,
    git_commit,
    load_py,
    read_json,
    rel,
    sha256_file,
)
from kaushik.core.scheduler import Job, run_scheduler

nz = load_py(KAUSHIK / "noise" / "common.py")


# ----------------------------------------------------------------------------- state


def adapter_record(cfg: dict, k: int) -> dict | None:
    gdir = nz.group_dir(cfg)
    na = read_json(gdir / "noise_adapters.json")
    if na:
        for a in na["adapters"]:
            if int(a["k"]) == k:
                return a
    return read_json(gdir / "make" / f"noise{k}.json")


def kl_done(cfg: dict) -> bool:
    return all(nz.kl_record_path(cfg, e["run_name"]).exists() for e in cfg["group"]["entries"])


def seeds_ready(cfg: dict) -> bool:
    """All expected seeds registered, or the group waited `groups.wait_hours` since `watch`
    first saw it (runs/watch_state.json)."""
    g = cfg["group"]
    if len(g["entries"]) >= int(g["expected_seeds"]):
        return True
    st = read_json(nz.runs_dir(cfg) / "watch_state.json", {}) or {}
    first = ((st.get("groups") or {}).get(cfg["run_name"]) or {}).get("first_seen")
    return bool(first) and time.time() - float(first) >= float(cfg["groups"]["wait_hours"]) * 3600


def target_and_reference(cfg: dict) -> tuple[dict, float, dict]:
    """(KL per trained seed, median, the seed whose KL is closest to the median)."""
    entries = cfg["group"]["entries"]
    kls = {}
    for e in entries:
        rec = nz.kl_record(cfg, e["run_name"])
        if rec is None:
            raise FileNotFoundError(f"no KL record for {e['run_name']} ({rel(nz.kl_record_path(cfg, e['run_name']))})")
        kls[e["run_name"]] = rec["kl"]["kl_per_token"]
    target = statistics.median(kls.values())
    ref = min(entries, key=lambda e: abs(kls[e["run_name"]] - target))
    return kls, target, ref


def engine(cfg: dict):
    src = nz.source_config(cfg["group"]["entries"][0]["config_path"])
    return nz.kl_engine(cfg, src, cfg["group"]["eval_chat_template_kwargs"])


# ----------------------------------------------------------------------------- work


def write_manifest(cfg: dict) -> None:
    mg = nz.make_groups_module()
    g, gdir = cfg["group"], nz.group_dir(cfg)
    files = sorted(ROOT.glob("datasets/*.csv")) + [ROOT / "datasets/splits.json"]
    nz.write_json({
        "group": mg.group_record(cfg), "settings": {k: cfg[k] for k in ("kl", "noise", "replay")},
        "config": train_eval_config(cfg), "timestamp": nz.now(), "git_commit": git_commit(),
        "modal_volume": defaults()["modal"]["checkpoints_volume"],
        "datasets_sha256": {rel(p): sha256_file(p) for p in files if p.exists()},
        "source_configs_sha256": {e["config_path"]: sha256_file(ROOT / e["config_path"])
                                  for e in g["entries"] if (ROOT / e["config_path"]).exists()},
        "registry_entries": {e["run_name"]: read_json(ROOT / e["registry_path"]) for e in g["entries"]},
    }, gdir / "manifest.json")


async def make_seed(cfg: dict, k: int, seed: int) -> dict:
    from cotcontrol.inference import modal_vllm

    g, n, name = cfg["group"], cfg["noise"], cfg["run_name"]
    base, gdir, rl = g["base_model"], nz.group_dir(cfg), nz.random_lora(cfg)
    root, calib = n["out_root"], f"{n['out_root']}/_calib"
    gen = engine(cfg)
    async with rl.app.run(), modal_vllm.app.run():
        samples, base_lps = await nz.ensure_samples(cfg, base, gen)
        kls, target, ref = target_and_reference(cfg)
        if target <= 0:
            raise SystemExit(f"{name}: median trained KL {target} is not positive; nothing to match")
        scale0 = float(n["scale0"]) if k == 0 else float(adapter_record(cfg, 0)["scale"])
        print(f"[make] {name} noise{k} (seed {seed}): trained KL/token "
              + ", ".join(f"seed{e['seed']}={kls[e['run_name']]:.5f}" for e in g["entries"])
              + f"; target (median) {target:.5f}; shapes from {ref['run_name']}; scale0 {scale0:.4g}", flush=True)

        async def kl_at(scale: float) -> float:
            path = f"{calib}/s{seed}_x{scale:.6g}"
            await nz.build_adapter(cfg, ref["adapter_path"], path, scale, seed, n["family"])
            kl = (await nz.measure_kl(cfg, path, base, gen, samples, base_lps))["kl"]["kl_per_token"]
            print(f"[make] {name} seed{seed} scale {scale:.4g}: KL {kl:.5f} (target {target:.5f}, ratio {kl / target:.3f})",
                  flush=True)
            return kl

        scale, hist = await rl.calibrate_scale(kl_at, target, scale0=scale0, tol=float(n["tolerance"]),
                                               max_evals=int(n["max_evals"]), grow=float(n["grow"]))
        final = f"{root}/noise{k}"
        build = await nz.build_adapter(cfg, ref["adapter_path"], final, scale, seed, n["family"])
        res = await nz.measure_kl(cfg, final, base, gen, samples, base_lps)
        nz.write_kl_record(cfg, f"{name}--noise{k}", base, final, res, gen, {"group": name, "noise_seed": seed, "scale": scale})
        print(f"[make] {name} noise{k} (seed {seed}): scale {scale:.4g} KL {res['kl']['kl_per_token']:.5f} "
              f"after {len(hist)} calibration evals -> {final}", flush=True)
    rec = {"k": k, "seed": seed, "scale": scale, "path": final, "kl": res["kl"], "calibration": hist, "build": build}
    nz.write_json(rec, gdir / "make" / f"noise{k}.json")
    return rec


async def finish(cfg: dict) -> dict:
    from dataclasses import asdict

    g, n, name = cfg["group"], cfg["noise"], cfg["run_name"]
    kls, target, ref = target_and_reference(cfg)
    adapters = [adapter_record(cfg, k) for k in range(len(n["seeds"]))]
    record = {"group": name, "base_model": g["base_model"], "target_kl": target, "trained_kl": kls,
              "reference": {"run_name": ref["run_name"], "adapter_path": ref["adapter_path"]},
              "family": n["family"], "tolerance": n["tolerance"], "kl_settings": nz.kl_settings(cfg),
              "engine": asdict(engine(cfg)), "adapters": adapters, "timestamp": nz.now()}
    nz.write_json(record, nz.group_dir(cfg) / "noise_adapters.json")
    rl = nz.random_lora(cfg)
    async with rl.app.run():
        await rl.remote_fn("remove_path").remote.aio(f"{n['out_root']}/_calib")
    return record


# ----------------------------------------------------------------------------- jobs


def jobs(cfg: dict, force: bool = False) -> list[Job]:
    g = cfg["group"]
    if not g["noise"]:
        return []
    name, gdir = cfg["run_name"], nz.group_dir(cfg)
    rank = nz.model_rank(cfg, g["base_model"])
    tp = int(engine(cfg).tensor_parallel_size)
    na = gdir / "noise_adapters.json"

    async def manifest():
        write_manifest(cfg)

    out = [Job(name=f"manifest:{name}", weight=0, run=manifest, done=(gdir / "manifest.json").exists,
               priority=(2, rank, name), group=str(gdir))]
    for k, seed in enumerate(cfg["noise"]["seeds"]):
        out.append(Job(
            name=f"make:{name}/noise{k}", weight=tp, run=lambda k=k, s=seed: make_seed(cfg, k, s),
            done=(lambda: False) if force else (lambda k=k: adapter_record(cfg, k) is not None),
            ready=lambda k=k: kl_done(cfg) and seeds_ready(cfg) and (k == 0 or adapter_record(cfg, 0) is not None),
            priority=(3, rank, name, k), group=str(gdir)))
    out.append(Job(
        name=f"make:{name}/finish", weight=0, run=lambda: finish(cfg), done=(lambda: False) if force else na.exists,
        ready=lambda: all(adapter_record(cfg, k) is not None for k in range(len(cfg["noise"]["seeds"]))),
        priority=(4, rank, name), group=str(gdir)))
    return out


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="noise make", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default=None, help="one group of sweeps/groups.yaml (default: every noise group)")
    ap.add_argument("--force", action="store_true", help="rebuild even when noise_adapters.json exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    points = [nz.group_point(cfg, a.group)] if a.group else nz.group_points(cfg)
    all_jobs = [j for p in points for j in jobs(p, force=a.force)]
    if not all_jobs:
        print("[make] nothing to do (no noise group)")
        return
    run_scheduler(all_jobs, nz.runs_dir(cfg), cap=a.cap, name="make_scheduler", dry_run=a.dry_run)


def group_dir_of(cfg: dict) -> Path:
    return nz.group_dir(cfg)
