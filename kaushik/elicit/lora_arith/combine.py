"""Build one servable combined adapter on the checkpoints volume (arith_app.py `combine`, CPU):
delta = sum_i coeff_i * delta_i by rank concatenation, verified module by module before committing.

    run.py elicit/lora_arith combine run_name=gepa_m2.0b_s0 "arith.terms=[[/checkpoints/...,1],[/checkpoints/...,-2]]"
    run.py sweep elicit/lora_arith/sweeps/combos.yaml --dry-run

-> <arith.combos_dir>/<run_name> on the volume; record in runs/_combos/<run_name>.json (terms, ranks, errors).
"""

from __future__ import annotations

import argparse
import json

from kaushik.core.io import atomic_write_json
from kaushik.core.scheduler import Job, run_scheduler
from kaushik.elicit.common import app_name, check_modal, runs_dir, setup_modal


def record_path(cfg: dict):
    return runs_dir(cfg) / "_combos" / f"{cfg['run_name']}.json"


def terms_of(cfg: dict) -> list[tuple[str, float]]:
    terms = [(str(p), float(c)) for p, c in cfg["arith"]["terms"]]
    if not terms:
        raise ValueError("arith.terms is empty")
    return terms


async def run(cfg: dict) -> dict:
    import modal

    dst = f"{cfg['arith']['combos_dir']}/{cfg['run_name']}"
    fn = modal.Function.from_name(app_name(cfg, "arith"), "combine")
    res = await fn.remote.aio(terms_of(cfg), dst, bool(cfg["arith"]["overwrite"]))
    atomic_write_json(res, record_path(cfg), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "terms"}, indent=1))
    print(f"-> {dst}  (max_rel_err {res['max_rel_err']:.2e})", flush=True)
    return res


def jobs(cfg: dict) -> list[Job]:
    setup_modal(cfg)
    rec = record_path(cfg)
    return [Job(name=f"combine:{cfg['run_name']}", weight=0, run=lambda: run(cfg), done=rec.exists,
                priority=(0, cfg["run_name"]), group=str(rec.parent / cfg["run_name"]))]


def main(cfg: dict, argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="combine", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    if a.dry_run:
        print(f"combine {cfg['run_name']}: {terms_of(cfg)} -> {cfg['arith']['combos_dir']}/{cfg['run_name']}")
    else:
        check_modal(cfg)
    run_scheduler(jobs(cfg), runs_dir(cfg), cap=a.cap, name="combine_scheduler", dry_run=a.dry_run)
