"""KL(base || adapter) per token for the registered adapters of one base model: a point of
sweeps/kl.yaml (`kl.base_model`, `kl.chat_template_kwargs`, `kl.engine_config`, `kl.entries`).

    .venv/bin/python kaushik/run.py sweep noise/sweeps/kl.yaml --dry-run
    .venv/bin/python kaushik/run.py noise kl [--runs NAME ...] [--model BASE] [--dry-run]
    .venv/bin/python kaushik/run.py noise kl --adapter /checkpoints/... --base-model M [--template-kwargs JSON] [--config CFG] [--name N]

Jobs per point: `samples:<tag>` (the base model's T=1 samples on `kl.n_prompts` unconstrained
val questions + its own teacher-forced log-probs, runs/_samples/<tag>/), then one `kl:<run>`
per adapter (teacher-forced log-probs of the same tokens; mean(log p_base - log p_adapter) is
an unbiased per-token KL estimate with a prompt-level bootstrap SE) -> runs/_kl/<run>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kaushik.core.io import KAUSHIK, load_py, read_json
from kaushik.core.scheduler import Job, run_scheduler

nz = load_py(KAUSHIK / "noise" / "common.py")


def engine(cfg: dict, entry: dict | None = None):
    """The point's engine (`kl.engine_config`: samples + base log-probs), or an adapter's own
    source run's engine when `entry` names a config (falls back to the point's)."""
    path = (entry or {}).get("config_path") or cfg["kl"]["engine_config"]
    src = nz.source_config(path) or nz.source_config(cfg["kl"]["engine_config"])
    return nz.kl_engine(cfg, src, cfg["kl"]["chat_template_kwargs"])


def jobs(cfg: dict, force: bool = False) -> list[Job]:
    k = cfg["kl"]
    base, kw = k["base_model"], k["chat_template_kwargs"]
    if not base or not k["entries"]:
        return []
    gen = engine(cfg)
    tp = int(gen.tensor_parallel_size)
    d = nz.samples_dir(cfg, base, kw)
    rank, tag = nz.model_rank(cfg, base), d.name

    async def samples():
        await nz.ensure_samples(cfg, base, gen)

    out = [Job(name=f"samples:{tag}", weight=tp, run=samples, done=lambda: (d / "base_logprobs.json").exists(),
               priority=(0, rank, tag))]

    def kl_job(e: dict) -> Job:
        rec = nz.kl_record_path(cfg, e["run_name"])
        egen = engine(cfg, e)

        async def run():
            t0 = time.time()
            samples, base_lps = await nz.ensure_samples(cfg, base, gen)
            res = await nz.measure_kl(cfg, e["adapter_path"], base, egen, samples, base_lps)
            nz.write_kl_record(cfg, e["run_name"], base, e["adapter_path"], res, egen,
                               {"experiment": e.get("experiment"), "arm": e.get("arm"), "seed": e.get("seed")})
            r = res["kl"]
            print(f"[kl] {e['run_name']}: KL/token {r['kl_per_token']:.5f} ± {r['se_prompt']:.5f} "
                  f"(prompt bootstrap; token SE {r['se_token']:.5f}; {r['n_tokens']} tokens) {time.time() - t0:.0f}s",
                  flush=True)

        return Job(name=f"kl:{e['run_name']}", weight=int(egen.tensor_parallel_size), run=run,
                   done=(lambda: False) if force else rec.exists,
                   ready=lambda: (d / "base_logprobs.json").exists(), priority=(1, rank, tag, e["run_name"]))

    return out + [kl_job(e) for e in k["entries"]]


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="noise kl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="registry run names (default: every pending adapter)")
    ap.add_argument("--model", default=None, help="only this base model")
    ap.add_argument("--adapter", default=None, help="ad hoc: /checkpoints/... adapter dir")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--template-kwargs", default=None, help="JSON, e.g. '{\"reasoning_effort\": \"medium\"}'")
    ap.add_argument("--config", default=None, help="train_eval config for engine params (ad hoc)")
    ap.add_argument("--name", default=None, help="record name (ad hoc)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    if a.adapter:
        if not a.base_model:
            raise SystemExit("--adapter needs --base-model")
        cfg = dict(cfg, kl={**cfg["kl"], "base_model": a.base_model, "engine_config": a.config,
                            "chat_template_kwargs": json.loads(a.template_kwargs) if a.template_kwargs else None,
                            "entries": [{"run_name": a.name or Path(a.adapter.rstrip("/")).parent.name,
                                         "adapter_path": a.adapter, "config_path": a.config or "",
                                         "experiment": None, "arm": None, "seed": None}]})
        points = [cfg]
    elif cfg["kl"]["entries"]:
        points = [cfg]
    else:
        points = nz.kl_points(cfg)
    all_jobs = []
    for p in points:
        if a.model and p["kl"]["base_model"] != a.model:
            continue
        if a.runs is not None:
            p = dict(p, kl={**p["kl"], "entries": [e for e in p["kl"]["entries"] if e["run_name"] in set(a.runs)]})
        all_jobs += jobs(p, force=a.force)
    if not all_jobs:
        print("[kl] nothing to do")
        return
    run_scheduler(all_jobs, nz.runs_dir(cfg), cap=a.cap, name="kl_scheduler", dry_run=a.dry_run)


def records(cfg: dict) -> dict[str, dict]:
    """run_name -> KL record for every adapter of this point (missing ones absent)."""
    return {e["run_name"]: r for e in cfg["kl"]["entries"] if (r := read_json(nz.kl_record_path(cfg, e["run_name"])))}
