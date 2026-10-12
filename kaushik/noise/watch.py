"""Poll the registry and process groups as they complete, under the shared GPU scheduler.

    .venv/bin/python kaushik/run.py noise watch [--interval S] [--max-hours H] [--idle-hours H] [--cap N] [--once] [--dry-run]

Every poll: make_groups rewrites sweeps/{groups,kl}.yaml from kaushik/registry, the KL jobs of
every base model and the make -> replay -> analyze jobs of every noise group are built, and one
scheduler pass runs everything that is ready under the cap (live override: runs/.gpu_cap;
`.stop_launches` pauses; the 800-container and memory guards are the scheduler's). A group's
`make` becomes ready once its KL records exist and its seeds are complete, or
`groups.wait_hours` after this watcher first saw it (runs/watch_state.json). The pass returns
when nothing more is ready; the loop sleeps `watch.interval_s` and polls again, writing
runs/progress.md, until `watch.max_hours` elapse or `watch.idle_hours` pass with no new entry
once every known group is processed.
"""


from __future__ import annotations

import argparse
import time
from pathlib import Path

from kaushik.core.config import defaults
from kaushik.core.io import KAUSHIK, load_py, read_json, rel
from kaushik.core.scheduler import run_scheduler

nz = load_py(KAUSHIK / "noise" / "common.py")


def _mods():
    return [load_py(KAUSHIK / "noise" / f"{m}.py") for m in ("make", "replay", "analyze")]


def all_jobs(cfg: dict, kpts: list[dict], gpts: list[dict]) -> list:
    klm = load_py(KAUSHIK / "noise" / "kl.py")
    jobs = [j for c in kpts for j in klm.jobs(c)]
    for c in gpts:
        for m in _mods():
            jobs += m.jobs(c)
    return jobs


def group_state(cfg: dict) -> str:
    gdir, g = nz.group_dir(cfg), cfg["group"]
    if not g["noise"]:
        return "KL only"
    if (gdir / "results.json").exists():
        return "done"
    if (gdir / "eval_summary.json").exists():
        return "evaluated"
    if (gdir / "noise_adapters.json").exists():
        n = len(list((gdir / "eval").glob("*/noise*.json"))) if (gdir / "eval").is_dir() else 0
        return f"adapters built, {n} evals"
    return "pending"


def write_progress(cfg: dict, gpts: list[dict], kpts: list[dict], state: dict, summary: dict, missing: dict) -> Path:
    runs = nz.runs_dir(cfg)
    pend = summary.get("pending", [])
    L = [f"# noise watch progress — {nz.now()}\n",
         (f"started {state.get('started')}; last pass: {len(summary.get('done', []))} jobs done, "
          f"{len(summary.get('failed', []))} failed, {len(pend)} pending ({', '.join(pend[:6])}{'...' if len(pend) > 6 else ''})\n"),
         "| group | seeds (have/expected) | KL measured | state | first seen |", "|---|---|---|---|---|"]
    for c in gpts:
        g, name = c["group"], c["run_name"]
        n_kl = sum(nz.kl_record_path(c, e["run_name"]).exists() for e in g["entries"])
        L.append(f"| {name} | {len(g['entries'])}/{g['expected_seeds']} | {n_kl}/{len(g['entries'])} | "
                 f"{group_state(c)} | {state['groups'].get(name, {}).get('first_seen_str', '')} |")
    L += ["\n## KL(base || adapter) per token\n", "| run | KL | ±se (prompt bootstrap) | tokens |", "|---|---|---|---|"]
    for c in kpts:
        for e in sorted(c["kl"]["entries"], key=lambda e: e["run_name"]):
            rec = nz.kl_record(c, e["run_name"])
            if rec:
                k = rec["kl"]
                L.append(f"| {e['run_name']} | {k['kl_per_token']:.5f} | {k['se_prompt']:.5f} | {k['n_tokens']} |")
            else:
                L.append(f"| {e['run_name']} | pending | | |")
    if missing:
        L += ["\n## Configs without a registry entry yet\n"] + [f"- {r} ({exp})" for r, exp in sorted(missing.items())]
    if summary.get("failed"):
        L += ["\n## Failed jobs (see watch.log)\n"] + [f"- {j}" for j in summary["failed"]]
    path = runs / "progress.md"
    path.write_text("\n".join(L) + "\n")
    return path


def main(cfg: dict, argv: list[str] | None = None) -> None:
    w = cfg["watch"]
    ap = argparse.ArgumentParser(prog="noise watch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=float(w["interval_s"]), help="seconds between polls")
    ap.add_argument("--max-hours", type=float, default=float(w["max_hours"]))
    ap.add_argument("--idle-hours", type=float, default=float(w["idle_hours"]))
    ap.add_argument("--cap", type=int, default=None, help="GPU cap (default defaults.yaml scheduler.max_gpus; live: runs/.gpu_cap)")
    ap.add_argument("--once", action="store_true", help="one poll + one scheduler pass")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; launch nothing, write no state")
    a = ap.parse_args(argv)
    mg = nz.make_groups_module()
    runs = nz.runs_dir(cfg)
    runs.mkdir(parents=True, exist_ok=True)
    state_path = runs / "watch_state.json"
    state = read_json(state_path, {}) or {}
    state.setdefault("groups", {})
    state["started"] = state.get("started") or nz.now()
    known: set[str] = set()
    t_start = last_new = time.time()
    while True:
        now = time.time()
        entries = mg.load_entries()
        mg.write_sweeps(cfg, entries)
        new = {e["run_name"] for e in entries} - known
        if new:
            known |= new
            last_new = now
            print(f"[watch] {nz.now()} registry: {len(entries)} entries ({len(new)} new)", flush=True)
        gpts, kpts = nz.group_points(cfg), nz.kl_points(cfg)
        for c in gpts:
            state["groups"].setdefault(c["run_name"], {"first_seen": now, "first_seen_str": nz.now()})
        if not a.dry_run:
            nz.write_json(state, state_path)
        settings = {**defaults()["scheduler"], "idle_exit_minutes": a.interval / 60}
        summary = run_scheduler(all_jobs(cfg, kpts, gpts), runs, cap=a.cap, name="watch", dry_run=a.dry_run,
                                settings=settings)
        missing = {r: x for r, x in mg.expected_run_names(cfg).items() if r not in known}
        if not a.dry_run:
            print(f"[watch] wrote {rel(write_progress(cfg, gpts, kpts, state, summary, missing))}", flush=True)
        if a.once or a.dry_run:
            return
        all_done = not summary["pending"] and not summary["failed"]
        elapsed_h, idle_h = (time.time() - t_start) / 3600, (time.time() - last_new) / 3600
        if elapsed_h >= a.max_hours:
            print(f"[watch] stopping: {elapsed_h:.1f}h elapsed", flush=True)
            return
        if idle_h >= a.idle_hours and all_done and known:
            print(f"[watch] stopping: no new entries for {idle_h:.1f}h and every registered run is processed "
                  f"({len(missing)} configs never registered)", flush=True)
            return
        time.sleep(a.interval)
