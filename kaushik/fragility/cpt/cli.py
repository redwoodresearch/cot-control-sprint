"""Stages for Experiment 1 arms 1B (cpt) / 1C (cpt_param_sweep).

    .venv/bin/python kaushik/run.py fragility/cpt <stage> [args]

  data       build data/c4_cpt.jsonl (+ smoke file + manifest with sha256)
  configs    write configs/*.yaml (make_configs.py)
  smoke      4 tiny end-to-end runs (2 steps, 8 questions/block) through train_eval.py
  train      enqueue stage 1 (1C seed-0 LR sweep, then 1B; both models) and start the scheduler
  scheduler  cap-aware job runner (foreground loop; `train` starts it in the background)
  select     pick each (model, k)'s LR on val, enqueue seeds 1-2 + the seed-0 test eval
  status     queue + per-run progress
  registry   write kaushik/registry/<run>.json for every finished final adapter
  analyze    tables + plots -> results/ (analyze.py)

GPU budget. Training and eval are separate jobs so the sprint's cap on
concurrent GPUs (40 for this agent, half of that 6 h after the first launch)
is spent on 1-GPU training jobs first and eval engines (EVAL_GPUS per run) as
capacity frees. The scheduler reserves each job's footprint for its lifetime
plus a short cooldown, never admits past the cap, and logs the workspace-wide
`modal container list` count for cross-checking. A training job runs
train_eval.py on a copy of the config with every eval block disabled; the
eval job later restores the canonical eval section into the run's config.yaml
and runs `train_eval.py --eval-only`, so run folders end up exactly as
train_eval.py normally leaves them (train.log, eval.log, eval/<block>/
checkpoint-<step>.json, eval_summary.json, summary.png).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import make_configs as mc

RUNS = HERE / "runs"
QUEUE = RUNS / "_queue"
RESULTS = HERE / "results"
REGISTRY = ROOT / "kaushik" / "registry"
TRAIN_EVAL = ROOT / "scripts" / "train_eval.py"
PY = sys.executable
DEFAULT_CAP = 90          # this agent's cap on concurrent Modal GPUs (coordinator, 2026-10-08 12:20: 120 -> 90)
WORKSPACE_CONTAINER_MAX = 800   # guard 1: launch nothing while the whole workspace runs more
MIN_FREE_MEM_PCT = 25     # guard 2: no new local driver processes below this free-memory share
COOLDOWN_S = 180          # a finished job's GPUs still count this long (container teardown)
MAX_ATTEMPTS = {"train": 4, "eval": 100}  # automatic re-queue after a failed attempt (evals resume from cache)
POLL_S = 30
GUARD = 0.85              # LR selection: final accuracy must keep >= 85% of checkpoint-0 accuracy
MAX_SWEEP_CELLS = 5       # 3 base rungs + up to 2 added below them when all fail the guard
TIE_PTS = 0.01            # honest-compliance differences up to this count as ties (-> higher LR)


# ----------------------------------------------------------------------------- utils

def now() -> float:
    return time.time()


def ts(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t or now()))


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def dump_yaml(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(d, sort_keys=False, width=120))


def atomic_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.rename(p)


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10, check=False).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def data_manifest() -> dict:
    return read_json(HERE / "data" / "c4_cpt.manifest.json", {}) or {}


def cfg_path(run_name: str) -> Path:
    return mc.CONFIG_DIR / f"{run_name}.yaml"


def run_dir(run_name: str) -> Path:
    return RUNS / run_name


# ----------------------------------------------------------------------------- jobs

def job_path(job_id: str) -> Path:
    return QUEUE / f"{job_id}.json"


def load_jobs() -> list[dict]:
    QUEUE.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in (read_json(p) for p in QUEUE.glob("*.json")
                        if p.name not in ("scheduler.json",)) if j]
    return sorted(jobs, key=lambda j: (j["priority"], j["created_at"]))


def save_job(j: dict) -> None:
    atomic_json(job_path(j["id"]), j)


def enqueue(kind: str, run_name: str, priority: int, suffix: str = "") -> dict | None:
    """Add a pending job unless one with this id exists. Footprint: 1 GPU for
    training, the config's eval engine count for eval."""
    jid = f"{kind}-{run_name}{suffix}"
    if job_path(jid).exists():
        return None
    cfg = load_yaml(cfg_path(run_name))
    job = {
        "id": jid, "kind": kind, "run_name": run_name, "priority": priority,
        "config": str(cfg_path(run_name).relative_to(ROOT)),
        "run_dir": str(run_dir(run_name).relative_to(ROOT)),
        "footprint": 1 if kind == "train" else mc.eval_footprint(cfg),
        "status": "pending", "pid": None, "started_at": None, "ended_at": None,
        "exit_code": None, "created_at": now(),
    }
    save_job(job)
    return job


def write_manifest(run_name: str, cfg: dict) -> None:
    dm = data_manifest()
    rd = run_dir(run_name)
    rd.mkdir(parents=True, exist_ok=True)
    atomic_json(rd / "manifest.json", {
        "run_name": run_name, **cfg["cpt_meta"], "base_model": cfg["base_model"],
        "config_path": str(cfg_path(run_name).relative_to(ROOT)),
        "data_path": cfg["train"]["data_path"],
        "data_sha256": {mc.DATA: dm.get("sha256"), mc.DATA_50: dm.get("sha256_50"),
                        mc.SMOKE_DATA: dm.get("smoke_sha256")}.get(cfg["train"]["data_path"]),
        "data_rows": {mc.DATA: dm.get("n_docs"), mc.DATA_50: dm.get("n_docs_50"),
                      mc.SMOKE_DATA: dm.get("smoke_docs")}.get(cfg["train"]["data_path"]),
        "data_manifest": f"{mc.REL}/data/c4_cpt.manifest.json",
        "max_seq_length": cfg["train"]["max_seq_length"], "batch_size": cfg["train"]["batch_size"],
        "max_steps": cfg["train"]["max_steps"], "save_at_steps": cfg["train"]["save_at_steps"],
        "lora": cfg["lora"], "elicitation": cfg["elicitation"],
        "eval_blocks": {n: {k: b[k] for k in ("split", "mode", "temperature", "max_tokens", "eval_at_steps")
                            if k in b}
                        for n, b in cfg["eval"].items()},
        "git_commit": git_commit(), "launched_at": ts(),
    })


def launch(job: dict) -> subprocess.Popen:
    rn = job["run_name"]
    canonical = load_yaml(cfg_path(rn))
    rd = run_dir(rn)
    rd.mkdir(parents=True, exist_ok=True)
    if job["kind"] == "train":
        train_cfg = copy.deepcopy(canonical)
        for b in train_cfg["eval"].values():
            b["run"] = False
        tc_path = rd / "train_config.yaml"
        dump_yaml(tc_path, train_cfg)
        write_manifest(rn, canonical)
        cmd, log = [PY, str(TRAIN_EVAL), str(tc_path)], rd / "launch.log"
    else:
        run_cfg = load_yaml(rd / "config.yaml")  # written by train_eval.py after training
        run_cfg["eval"] = canonical["eval"]
        dump_yaml(rd / "config.yaml", run_cfg)
        job["footprint"] = mc.eval_footprint(canonical)  # the config may have changed since enqueue
        cmd, log = [PY, str(TRAIN_EVAL), "--eval-only", str(rd)], rd / "eval_launch.log"
    with open(log, "a") as fh:  # the child keeps its own copy of the fd
        fh.write(f"\n==== {ts()} {job['id']}: {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
    job.update(status="running", pid=proc.pid, started_at=now())
    save_job(job)
    return proc


def artifact_ok(job: dict) -> bool:
    rd = ROOT / job["run_dir"]
    if job["kind"] == "train":
        return (rd / "train_result.json").exists()
    es = rd / "eval_summary.json"
    return es.exists() and es.stat().st_mtime >= (job["started_at"] or 0)


# ----------------------------------------------------------------------------- scheduler

def modal_container_count() -> int | None:
    """Workspace-wide running containers (all agents), best effort."""
    try:
        out = subprocess.run(["modal", "container", "list", "--json"], capture_output=True,
                             text=True, timeout=60, check=False,
                             env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"})
        if out.returncode == 0:  # rich may still colorize: strip ANSI escapes first
            return len(json.loads(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", out.stdout)))
    except Exception as e:  # noqa: BLE001
        print(f"[scheduler] modal container list failed: {e!r}", file=sys.stderr)
    return None


def current_cap(base: int) -> int:
    """GPU cap; runs/_queue/cap.txt overrides it live."""
    override = QUEUE / "cap.txt"
    if override.exists():
        try:
            return int(override.read_text().strip())
        except ValueError:
            pass
    return base


def free_memory_pct() -> int | None:
    """macOS `memory_pressure` -> system-wide free memory percentage."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30, check=False)
        m = re.search(r"free percentage:\s*(\d+)%", out.stdout)
        return int(m.group(1)) if m else None
    except Exception as e:  # noqa: BLE001
        print(f"[scheduler] memory_pressure failed: {e!r}", file=sys.stderr)
        return None


def guards_ok(containers: int | None, free_mem: int | None) -> tuple[bool, str]:
    """Admission guards from the coordinator: workspace-wide container count and
    laptop free memory. Unknown values (None) do not block."""
    if containers is not None and containers > WORKSPACE_CONTAINER_MAX:
        return False, f"workspace containers {containers} > {WORKSPACE_CONTAINER_MAX}"
    if free_mem is not None and free_mem < MIN_FREE_MEM_PCT:
        return False, f"free memory {free_mem}% < {MIN_FREE_MEM_PCT}%"
    return True, ""


def scheduler(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="cap-aware job runner")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--poll", type=int, default=POLL_S)
    ap.add_argument("--exit-when-idle", action="store_true")
    args = ap.parse_args(argv)

    QUEUE.mkdir(parents=True, exist_ok=True)
    state_path = QUEUE / "scheduler.json"
    state = read_json(state_path, {}) or {}
    if pid_alive(state.get("pid")) and state.get("pid") != os.getpid():
        raise SystemExit(f"scheduler already running (pid {state['pid']})")
    state.update(pid=os.getpid(), base_cap=args.cap, started_at=now())
    atomic_json(state_path, state)
    log = open(QUEUE / "scheduler.log", "a")  # noqa: SIM115  (lives as long as the loop)
    procs: dict[str, subprocess.Popen] = {}
    last_count, last_count_t = None, 0.0

    def emit(msg: str) -> None:
        log.write(f"{ts()} {msg}\n")
        log.flush()

    emit(f"scheduler start pid={os.getpid()} cap={args.cap}")
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, _stop)

    while not stop:
        jobs = load_jobs()
        for j in jobs:
            if j["status"] != "running":
                continue
            proc = procs.get(j["id"])
            if proc is not None:
                rc = proc.poll()
                if rc is None:
                    continue
                ok = artifact_ok(j)  # the artifact exists only if training/eval completed (or was adopted)
                j["exit_code"] = rc
                procs.pop(j["id"])
            elif pid_alive(j["pid"]):
                continue  # adopted from a previous scheduler process
            else:
                ok = artifact_ok(j)
            j.update(status="done" if ok else "failed", ended_at=now())
            if not ok and j.get("attempts", 1) < MAX_ATTEMPTS[j["kind"]]:
                # Transient Modal/gRPC drops ("Connection lost") kill the local
                # driver; a re-run resumes evals from the generation cache.
                j.update(status="pending", pid=None, started_at=None, exit_code=None, attempts=j.get("attempts", 1) + 1)
            save_job(j)
            emit(f"{j['id']} -> {j['status']} (rc={j['exit_code']}"
                 + (f", retry {j['attempts']}/{MAX_ATTEMPTS[j['kind']]}" if j["status"] == "pending" else "") + ")")
            if ok and j["kind"] == "train":
                ej = enqueue("eval", j["run_name"], j["priority"])
                if ej:
                    emit(f"enqueued {ej['id']} footprint={ej['footprint']}")

        jobs = load_jobs()
        reserved = sum(j["footprint"] for j in jobs if j["status"] == "running")
        reserved += sum(j["footprint"] for j in jobs
                        if j["status"] in ("done", "failed") and j["ended_at"]
                        and now() - j["ended_at"] < COOLDOWN_S)
        cap = current_cap(args.cap)
        # One eval process per run at a time: a backfill (-v2) and a -test eval of
        # the same run would otherwise generate the same prompts twice and race
        # on the result files.
        busy_runs = {j["run_name"] for j in jobs if j["status"] == "running" and j["kind"] == "eval"}
        admissible = [j for j in jobs if j["status"] == "pending"
                      and (j["kind"] == "train" or ((ROOT / j["run_dir"] / "config.yaml").exists()
                                                    and j["run_name"] not in busy_runs))
                      and reserved + j["footprint"] <= cap]
        if admissible and now() - last_count_t > 60:
            last_count, last_count_t = modal_container_count(), now()
        free_mem = free_memory_pct() if admissible else None
        ok, why = guards_ok(last_count, free_mem)
        if admissible and not ok:
            emit(f"admission paused: {why}")
        for j in jobs:
            if j["status"] == "pending" and j["footprint"] > cap:
                emit(f"WARNING {j['id']} needs {j['footprint']} GPUs > cap {cap}; it will never start")
        for j in admissible if ok else []:
            if reserved + j["footprint"] > cap:
                continue
            try:
                procs[j["id"]] = launch(j)
            except Exception as e:  # noqa: BLE001
                j.update(status="failed", ended_at=now(), exit_code=None, error=repr(e))
                save_job(j)
                emit(f"{j['id']} failed to launch: {e!r}")
                continue
            reserved += j["footprint"]
            state.setdefault("first_launch_at", now())
            emit(f"launched {j['id']} pid={j['pid']} footprint={j['footprint']} reserved={reserved}/{cap}")

        if now() - last_count_t > 300:
            last_count, last_count_t = modal_container_count(), now()
        counts = {s: sum(j["status"] == s for j in jobs) for s in ("pending", "running", "done", "failed")}
        state.update(last_poll=now(), cap=cap, reserved=reserved, counts=counts,
                     workspace_containers=last_count, free_memory_pct=free_mem)
        atomic_json(state_path, state)
        emit(f"poll cap={cap} reserved={reserved} {counts} workspace_containers={last_count} free_mem={free_mem}")
        if args.exit_when_idle and not counts["pending"] and not counts["running"]:
            break
        time.sleep(args.poll)
    emit("scheduler exit")


def ensure_scheduler(cap: int) -> None:
    state = read_json(QUEUE / "scheduler.json", {}) or {}
    if pid_alive(state.get("pid")):
        print(f"scheduler running (pid {state['pid']})")
        return
    QUEUE.mkdir(parents=True, exist_ok=True)
    with open(QUEUE / "scheduler.out", "a") as out:
        proc = subprocess.Popen([PY, str(ROOT / "kaushik" / "run.py"), "fragility/cpt", "scheduler",
                                 "--cap", str(cap)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
    print(f"started scheduler pid={proc.pid} cap={cap} (log: {QUEUE.relative_to(ROOT)}/scheduler.log)")


# ----------------------------------------------------------------------------- stages

def data(argv: list[str]) -> None:
    import build_data
    build_data.main(argv)


def configs(argv: list[str]) -> None:
    mc.main(argv)


def smoke(argv: list[str]) -> None:
    """Integrated train+eval of the smoke configs (2 steps, 8 questions), then
    checks: adapter size vs formula, mask verification, eval files, honest
    compliance. Footprint <= 1 + EVAL_GPUS per run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(mc.MODELS))
    ap.add_argument("--timeout", type=int, default=3 * 3600)
    args = ap.parse_args(argv)
    cfgs = [c for c in mc.smoke_configs() if c["cpt_meta"]["model"] in args.models]
    procs = []
    for c in cfgs:
        p = mc.write(c)
        rd = run_dir(c["run_name"])
        rd.mkdir(parents=True, exist_ok=True)
        write_manifest(c["run_name"], c)
        with open(rd / "launch.log", "a") as fh:
            procs.append((c, subprocess.Popen([PY, str(TRAIN_EVAL), str(p)], cwd=ROOT, stdout=fh,
                                              stderr=subprocess.STDOUT, start_new_session=True)))
        print(f"launched {c['run_name']} pid={procs[-1][1].pid}")
    t0 = now()
    for c, proc in procs:
        try:
            proc.wait(timeout=max(1, int(args.timeout - (now() - t0))))
        except subprocess.TimeoutExpired:
            print(f"{c['run_name']}: TIMEOUT")
    import analyze
    ok = True
    for c, proc in procs:
        rd = run_dir(c["run_name"])
        tr = read_json(rd / "train_result.json")
        print(f"\n== {c['run_name']} rc={proc.returncode}")
        if not tr:
            print("   no train_result.json — see", rd / "launch.log")
            ok = False
            continue
        expect = mc.adapter_numel(c["cpt_meta"]["model"], c["cpt_meta"]["rank"])
        print(f"   total_lora_params={tr['total_lora_params']:,} formula={expect:,} "
              f"{'OK' if tr['total_lora_params'] == expect else 'MISMATCH'}; "
              f"steps={tr['num_steps']} final_loss={tr['final_loss']:.3f} "
              f"runtime={tr['train_runtime']}s ({tr['train_runtime'] / max(1, tr['num_steps']):.0f}s/step)")
        ok &= tr["total_lora_params"] == expect
        if tr["mask_verification"]:
            mv = tr["mask_verification"]
            print(f"   mask: selected={mv['n_selected']} changed_inside={mv['n_changed_inside_mask']} "
                  f"outside={mv['n_changed_outside_mask']}")
            ok &= mv["n_changed_outside_mask"] == 0
        for block in sorted((rd / "eval").glob("*")):
            for f in sorted(block.glob("checkpoint-*.json")):
                rows = analyze.load_rollouts(f)
                print(f"   {block.name}/{f.stem}: n={len(rows)} honest={analyze.mean(r['honest'] for r in rows):.2f} "
                      f"strict={analyze.mean(r['strict'] for r in rows):.2f} acc={analyze.mean(r['correct'] for r in rows):.2f} "
                      f"chars={analyze.mean(r['chars'] for r in rows):.0f} errors={sum(r['error'] for r in rows)}")
                ok &= len(rows) > 0 and not any(r["error"] for r in rows)
    print("\nSMOKE", "PASSED" if ok else "FAILED")


def train(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="enqueue stage-1 jobs and start the scheduler")
    ap.add_argument("--models", nargs="+", default=list(mc.MODELS), choices=list(mc.MODELS))
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-scheduler", action="store_true")
    args = ap.parse_args(argv)
    if not (HERE / "data" / "c4_cpt.jsonl").exists():
        raise SystemExit("no data/c4_cpt.jsonl — run the `data` stage first")
    cfgs = mc.stage1_configs(args.models)
    print(f"stage 1: {len(cfgs)} runs "
          f"({sum(c['cpt_meta']['arm'] == 'cpt' for c in cfgs)} x 1B, "
          f"{sum(c['cpt_meta']['arm'] != 'cpt' for c in cfgs)} x 1C seed-0 cells)")
    if args.dry_run:
        for c in cfgs:
            print(f"  {c['run_name']:<40} steps={c['train']['max_steps']} eval_gpus={mc.eval_footprint(c)}")
        return
    for c in cfgs:
        mc.write(c)
        j = enqueue("train", c["run_name"], 0 if c["cpt_meta"]["arm"] == "cpt_param_sweep" else 1)
        print(f"  {'queued ' if j else 'exists '} train-{c['run_name']}")
    if not args.no_scheduler:
        ensure_scheduler(args.cap)


def _val_metrics(rd: Path, step: int) -> dict | None:
    """Pooled honest/strict/accuracy over the val blocks at one step."""
    import analyze
    rows = []
    for m in mc.VAL_MODES:
        f = rd / "eval" / f"val_{m}" / f"checkpoint-{step}.json"
        if not f.exists():
            return None
        rows += analyze.load_rollouts(f)
    return {"n": len(rows), "honest": analyze.mean(r["honest"] for r in rows),
            "strict": analyze.mean(r["strict"] for r in rows),
            "accuracy": analyze.mean(r["correct"] for r in rows)}


def select(argv: list[str]) -> None:
    """Per (model, k): pick the seed-0 LR with the highest honest val compliance
    at step 50 among LRs whose accuracy kept >= GUARD x checkpoint-0 accuracy
    (ties -> lower LR; nothing passes -> least accuracy loss, flagged). Then
    add test blocks to the chosen run and enqueue its eval + seeds 1-2."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(mc.MODELS), choices=list(mc.MODELS))
    ap.add_argument("--ks", nargs="+", default=list(mc.KS), choices=list(mc.KS))
    ap.add_argument("--allow-partial", action="store_true", help="select among evaluated cells only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--guard", type=float, default=GUARD)
    args = ap.parse_args(argv)
    sel_path = RESULTS / "lr_selection.json"
    selection = read_json(sel_path, {}) or {}
    for m in args.models:
        for klabel in args.ks:
            key = f"{m}/{klabel}"
            if key in selection and not args.dry_run:
                print(f"{key}: already selected lr={selection[key]['lr']}")
                continue
            cells = []
            for lr in mc.lr_grid(m, klabel):
                rn = mc.run_name_1c(m, klabel, lr, 0)
                rd = run_dir(rn)
                tr = read_json(rd / "train_result.json")
                base, final = _val_metrics(rd, 0), _val_metrics(rd, mc.STEPS_1C)
                mv = (tr or {}).get("mask_verification") or {}
                cells.append({"run_name": rn, "lr": lr, "base": base, "final": final,
                              "k_effective": mv.get("n_changed_inside_mask"),
                              "final_loss": (tr or {}).get("final_loss")})
            done = [c for c in cells if c["base"] and c["final"]]
            if len(done) < len(cells) and not args.allow_partial:
                print(f"{key}: {len(done)}/{len(cells)} cells evaluated; waiting (--allow-partial to force)")
                continue
            if not done:
                print(f"{key}: nothing evaluated")
                continue
            for c in done:
                c["acc_ratio"] = c["final"]["accuracy"] / max(c["base"]["accuracy"], 1e-9)
                c["guard_ok"] = c["acc_ratio"] >= args.guard
            passing = [c for c in done if c["guard_ok"]]
            if not passing and len(cells) < MAX_SWEEP_CELLS and not args.dry_run:
                # Every rung is too hot for this model/k: add one a factor 3 lower
                # and decide once it is evaluated (up to MAX_SWEEP_CELLS rungs).
                lr_new = mc.extend_lr_grid(m, klabel)
                cfg_new = mc.sweep_config(m, klabel, lr_new, 0, with_test=False)
                mc.write(cfg_new)
                j = enqueue("train", cfg_new["run_name"], 0)
                print(f"{key}: all {len(done)} LRs fail the accuracy guard -> extending the sweep "
                      f"down to lr={lr_new:.1e} ({'queued' if j else 'exists'})")
                continue
            # Among guard-passing LRs: highest honest compliance; cells within
            # TIE_PTS of the best are tied (800 rollouts: SE ~1 pt) and the tie
            # goes to the higher LR, i.e. the larger dose that still preserved
            # accuracy. Nothing passing after MAX_SWEEP_CELLS rungs -> least
            # accuracy loss, flagged.
            if passing:
                top = max(c["final"]["honest"] for c in passing)
                best = max((c for c in passing if top - c["final"]["honest"] <= TIE_PTS), key=lambda c: c["lr"])
            else:
                best = max(done, key=lambda c: (c["acc_ratio"], -c["lr"]))
            rec = {"model": m, "klabel": klabel, "lr": best["lr"], "run_name": best["run_name"],
                   "guard": args.guard, "guard_failed_all": not passing, "cells": cells,
                   "selected_at": ts()}
            print(f"{key}: lr={best['lr']:.1e} honest={best['final']['honest']:.3f} "
                  f"acc {best['base']['accuracy']:.3f}->{best['final']['accuracy']:.3f}"
                  f"{'  [GUARD FAILED FOR ALL LRs]' if not passing else ''}")
            for c in done:
                print(f"    lr={c['lr']:.1e} honest={c['final']['honest']:.3f} strict={c['final']['strict']:.3f} "
                      f"acc_ratio={c['acc_ratio']:.2f} k_eff={c['k_effective']} {'*' if c is best else ''}")
            if args.dry_run:
                continue
            selection[key] = rec
            atomic_json(sel_path, selection)
            # seed-0 winner: add the test blocks to its canonical config, re-eval (val from cache)
            cfg = mc.sweep_config(m, klabel, best["lr"], 0, with_test=True)
            mc.write(cfg)
            j = enqueue("eval", best["run_name"], 0, suffix="-test")
            print(f"    {'queued' if j else 'exists'} eval-{best['run_name']}-test")
            for c2 in mc.stage2_configs(m, klabel, best["lr"]):
                mc.write(c2)
                j = enqueue("train", c2["run_name"], 0)
                print(f"    {'queued' if j else 'exists'} train-{c2['run_name']}")
    if not args.dry_run:
        ensure_scheduler(DEFAULT_CAP)


def _modal_ls(path: str) -> list[str]:
    """Entries of a path on this agent's checkpoints volume (best effort)."""
    vol = f"cotcontrol_{os.environ.get('COTCONTROL_MODAL_OWNER', 'kaushik-reddy')}_checkpoints"
    try:
        out = subprocess.run(["modal", "volume", "ls", vol, path, "--json"], capture_output=True, text=True,
                             timeout=60, check=False, env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"})
        if out.returncode != 0:
            return []
        rows = json.loads(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", out.stdout))
        return [Path(r.get("Filename") or r.get("filename") or r.get("path") or "").name for r in rows]
    except Exception:  # noqa: BLE001
        return []


def volume_progress(run_names: list[str]) -> dict[str, str]:
    """run_name -> 'ckpt N' from the checkpoints volume: the worker's stdout
    does not reach the local log (deployed mode), but every saved checkpoint
    does land on the volume. One listing for the volume root, then one per run."""
    from concurrent.futures import ThreadPoolExecutor

    dirs = _modal_ls("")
    latest = {}
    for rn in run_names:
        mine = sorted(d for d in dirs if re.fullmatch(rf"{re.escape(rn)}-\d{{8}}_\d{{6}}", d))
        if mine:
            latest[rn] = mine[-1]

    def one(item):
        rn, d = item
        entries = _modal_ls(d) or _modal_ls(d)  # one retry: the CLI rate-limits bursts
        steps = [int(m.group(1)) for e in entries if (m := re.fullmatch(r"checkpoint-(\d+)", e))]
        return rn, (f"ckpt {max(steps)}" if steps else "started" if entries else "?") + f" ({d[-15:]})"

    with ThreadPoolExecutor(3) as ex:
        return dict(ex.map(one, latest.items()))


def _train_progress(rd: Path, volume: dict[str, str] | None = None) -> str:
    log = rd / "train.log"
    if not log.exists():
        return "-"
    txt = log.read_text(errors="ignore")
    steps = re.findall(r"\[step (\d+)/(\d+)\] loss=([\d.eE+-]+)", txt)
    if (rd / "train_result.json").exists():
        return f"done loss={steps[-1][2][:6]}" if steps else "done"
    if steps:
        return f"step {steps[-1][0]}/{steps[-1][1]} loss={steps[-1][2][:6]}"
    last = txt.strip().splitlines()[-1] if txt.strip() else ""
    if "slot-retry" in last:
        return "slot-retry"
    return (volume or {}).get(rd.name, "starting")


def _eval_progress(rd: Path) -> str:
    cfg = read_json(rd / "manifest.json", {}) or {}
    blocks = cfg.get("eval_blocks", {})
    if not blocks:
        return "-"
    have = {b.name: len(list(b.glob("checkpoint-*.json"))) for b in (rd / "eval").glob("*")} if (rd / "eval").exists() else {}
    done = (rd / "eval_summary.json").exists()
    return ("summary " if done else "") + " ".join(f"{n.replace('val_', 'v:').replace('test_', 't:')}={have.get(n, 0)}" for n in blocks)


def status(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, default=0, help="show the last N log lines of failed jobs")
    args = ap.parse_args(argv)
    state = read_json(QUEUE / "scheduler.json", {}) or {}
    print(f"scheduler pid={state.get('pid')} alive={pid_alive(state.get('pid'))} cap={state.get('cap')} "
          f"reserved={state.get('reserved')} counts={state.get('counts')} "
          f"workspace_containers={state.get('workspace_containers')} free_mem={state.get('free_memory_pct')}% "
          f"last_poll={ts(state['last_poll']) if state.get('last_poll') else '-'}")
    if state.get("first_launch_at"):
        print(f"first launch {ts(state['first_launch_at'])} ({(now() - state['first_launch_at']) / 3600:.1f} h ago)")
    jobs = load_jobs()
    vol = volume_progress([j["run_name"] for j in jobs if j["kind"] == "train" and j["status"] == "running"])
    print(f"\n{'job':<52} {'status':<8} {'fp':>2} {'started':<17} progress")
    for j in jobs:
        rd = ROOT / j["run_dir"]
        prog = _train_progress(rd, vol) if j["kind"] == "train" else _eval_progress(rd)
        print(f"{j['id']:<52} {j['status']:<8} {j['footprint']:>2} "
              f"{ts(j['started_at']) if j['started_at'] else '-':<17} {prog}")
        if args.tail and j["status"] == "failed":
            log = rd / ("launch.log" if j["kind"] == "train" else "eval_launch.log")
            if log.exists():
                print("      " + "\n      ".join(log.read_text(errors="ignore").splitlines()[-args.tail:]))
    sel = read_json(RESULTS / "lr_selection.json", {}) or {}
    if sel:
        print("\nLR selection: " + ", ".join(f"{k}: {v['lr']:.1e}" for k, v in sel.items()))


def registry(argv: list[str]) -> None:
    """Registry entries for finished final adapters. 1B: checkpoint-200 (run
    name) and checkpoint-50 (run name + '-step50'), noise + KL. 1C: the
    selected-LR runs' checkpoint-50, KL only."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    sel = read_json(RESULTS / "lr_selection.json", {}) or {}
    selected_1c = {mc.run_name_1c(v["model"], v["klabel"], v["lr"], s)
                   for v in sel.values() for s in mc.SEEDS}
    n = 0
    for rd in sorted(p for p in RUNS.glob("cpt*") if p.is_dir()):
        tr, man = read_json(rd / "train_result.json"), read_json(rd / "manifest.json")
        if not tr or not man:  # the adapter is final once training ends; evals follow
            continue
        is_1b = man["arm"] == "cpt"  # noise controls only for the spec 1B arm; low-LR variants get KL only
        if man["arm"] not in ("cpt", "cpt_lowlr") and rd.name not in selected_1c:
            continue
        paths = {c["step"]: c["path"] for c in tr["checkpoints"]}
        finals = [(s, "" if s == max(man["final_steps"]) else f"-step{s}") for s in man["final_steps"]]
        for step, suffix in finals:
            if step not in paths:
                continue
            entry = {
                "run_name": rd.name + suffix, "experiment": "fragility/cpt", "arm": man["arm"],
                "base_model": man["base_model"], "seed": man["seed"], "adapter_path": paths[step],
                "config_path": man["config_path"],
                "eval_chat_template_kwargs": mc.MODELS[man["model"]]["eval_template_kwargs"],
                "noise": is_1b, "measure_kl": True,
                "noise_eval_blocks": f"{man['config_path']}#" + ",".join(mc.TEST_BLOCKS),
                "noise_eval_block_names": mc.TEST_BLOCKS, "step": step,
                "lora_rank": man["rank"], "train_params": man["elicitation"]["train_params"],
                "k_effective": (tr.get("mask_verification") or {}).get("n_changed_inside_mask"),
                "run_dir": str(rd.relative_to(ROOT)), "written_at": ts(),
            }
            dst = REGISTRY / f"{entry['run_name']}.json"
            old = read_json(dst, {}) or {}
            if {k: v for k, v in old.items() if k != "written_at"} == {k: v for k, v in entry.items() if k != "written_at"}:
                continue  # unchanged: don't touch the file the noise agent polls
            if not args.dry_run:
                atomic_json(dst, entry)
            n += 1
            print(f"{'would write' if args.dry_run else 'wrote'} registry/{entry['run_name']}.json -> {paths[step]}")
    print(f"{n} entries")


def _modal_get(remote: str, local: Path) -> bool:
    vol = f"cotcontrol_{os.environ.get('COTCONTROL_MODAL_OWNER', 'kaushik-reddy')}_checkpoints"
    local.parent.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["modal", "volume", "get", "--force", vol, remote, str(local)], capture_output=True,
                         text=True, timeout=120, check=False, env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"})
    return out.returncode == 0 and local.exists()


def adopt(argv: list[str]) -> None:
    """Adopt a finished orphan Modal training run into its run dir. When a local
    driver is killed, the deployed-mode worker keeps training and writes its
    checkpoints, losses.jsonl, training_data.json and mask_verification.json to
    /checkpoints/<run>-<ts>/ on the volume; this rebuilds train_result.json and
    config.yaml from those files (same schema as train_eval.py writes, plus an
    `adopted` note) so the scheduler marks the job done and evaluates it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, e.g. cpt-qwen8b-r32-seed0")
    ap.add_argument("--volume-dir", required=True, help="finished dir on the volume, e.g. cpt-qwen8b-r32-seed0-20261008_100347")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    from dataclasses import asdict

    from cotcontrol.training.config import TrainConfig, filter_dataclass_kwargs

    rd = run_dir(args.run)
    raw = load_yaml(rd / "train_config.yaml")
    flat = {k: v for s in raw.values() if isinstance(s, dict) and s is not raw.get("eval") for k, v in s.items()}
    flat |= {k: v for k, v in raw.items() if not isinstance(v, dict)}
    tc = TrainConfig(**filter_dataclass_kwargs(TrainConfig, flat))
    tc.run_name = args.volume_dir
    entries = _modal_ls(args.volume_dir)
    steps = sorted(int(m.group(1)) for e in entries if (m := re.fullmatch(r"checkpoint-(\d+)", e)))
    if not steps or steps[-1] < tc.max_steps:
        raise SystemExit(f"{args.volume_dir}: checkpoints {steps} do not reach max_steps={tc.max_steps}; not finished")
    local = rd / "adopted"
    files = {n: _modal_get(f"{args.volume_dir}/{n}", local / n)
             for n in ("losses.jsonl", "training_data.json", "mask_verification.json", "mask.json")}
    if not files["losses.jsonl"] or not files["training_data.json"]:
        raise SystemExit(f"could not fetch losses.jsonl/training_data.json from {args.volume_dir}: {files}")
    losses = [json.loads(l)["loss"] for l in (local / "losses.jsonl").read_text().splitlines() if l.strip()]
    td = json.loads((local / "training_data.json").read_text())
    mv = json.loads((local / "mask_verification.json").read_text()) if files["mask_verification.json"] else None
    mask = json.loads((local / "mask.json").read_text()) if files["mask.json"] else {}
    result = {
        "run_name": args.volume_dir, "checkpoint_path": f"/checkpoints/{args.volume_dir}",
        "checkpoints": [{"step": s, "path": f"/checkpoints/{args.volume_dir}/checkpoint-{s}"} for s in steps],
        "final_step": steps[-1], "num_steps": len(losses),
        "avg_loss": sum(losses) / len(losses) if losses else None, "final_loss": losses[-1] if losses else None,
        "train_runtime": None, "n_examples": td.get("n_examples"), "n_truncated": td.get("n_truncated"),
        "total_lora_params": mask.get("total_lora_params") or mc.adapter_numel(raw["cpt_meta"]["model"], raw["cpt_meta"]["rank"]),
        "mask_verification": ({k: mv[k] for k in ("n_selected", "n_changed_inside_mask", "n_changed_outside_mask")}
                              if mv else None),
        "config": asdict(tc),
        "adopted": {"volume_dir": args.volume_dir, "at": ts(),
                    "reason": "local driver killed externally; the deployed-mode worker finished on its own"},
    }
    print(f"{args.run}: adopt {args.volume_dir} steps={steps} final_loss={result['final_loss']} mask={result['mask_verification']}")
    if args.dry_run:
        return
    (rd / "train_result.json").write_text(json.dumps(result, indent=2))
    dump_yaml(rd / "config.yaml", raw)
    with open(rd / "launch.log", "a") as fh:
        fh.write(f"\n==== {ts()} adopted finished orphan worker output from /checkpoints/{args.volume_dir}\n")
    print(f"wrote {rd / 'train_result.json'} and config.yaml; the scheduler will mark the job done once its driver is gone")


def lowlr(argv: list[str]) -> None:
    """Supplementary 1B variant at lower LRs (arm cpt_lowlr, seed 0): the spec
    dose (r32, lr 1.5e-4) halves gpt-oss's accuracy within 50 steps without
    raising compliance, so this checks a non-destructive dose closer to SDF's
    C4 arm (lr 1e-5). Same data, steps, checkpoints and eval blocks as 1B."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gptoss20b"], choices=list(mc.MODELS))
    ap.add_argument("--lrs", nargs="+", type=float, default=[5e-5, 1.5e-5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    for m in args.models:
        for lr in args.lrs:
            for s in args.seeds:
                cfg = mc.cpt_config(m, s, lr=lr, arm="cpt_lowlr")
                if args.dry_run:
                    print(f"would queue {cfg['run_name']}")
                    continue
                mc.write(cfg)
                j = enqueue("train", cfg["run_name"], 2)
                print(f"{'queued' if j else 'exists'} train-{cfg['run_name']}")
    if not args.dry_run:
        ensure_scheduler(DEFAULT_CAP)


def backfill(argv: list[str]) -> None:
    """Re-eval runs whose finished eval lacks a block of the current canonical
    config (e.g. the later-added val_repeat_sentences): one `eval-<run>-v2` job
    per run; the blocks already done come from the generation cache."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    jobs = {j["id"]: j for j in load_jobs()}
    n = 0
    for rd in sorted(p for p in RUNS.glob("cpt*") if p.is_dir()):
        if not (rd / "eval_summary.json").exists() or not cfg_path(rd.name).exists():
            continue
        if any(j["run_name"] == rd.name and j["kind"] == "eval" and j["status"] in ("pending", "running")
               for j in jobs.values()):
            continue  # an eval of this run is in flight; it reads the current config anyway
        want = {n_ for n_, b in load_yaml(cfg_path(rd.name))["eval"].items() if b.get("run", True)}
        have = set(json.loads((rd / "eval_summary.json").read_text()).get("evals", {}))
        if want - have:
            n += 1
            if args.dry_run:
                print(f"would backfill {rd.name}: {sorted(want - have)}")
            else:
                j = enqueue("eval", rd.name, 1, suffix="-v2")
                print(f"{'queued' if j else 'exists'} eval-{rd.name}-v2 for {sorted(want - have)}")
    print(f"{n} runs need backfill")


def losses(argv: list[str]) -> None:
    """Fetch every finished run's per-step C4 loss log (losses.jsonl, written by
    the Modal worker next to its checkpoints) into runs/<run>/losses.jsonl so
    `analyze` can compare arms at matched C4 loss (functional size). Step 1's loss
    is the untrained model's (LoRA B starts at zero). Read-only volume gets, a few
    in parallel; already-fetched files are kept unless --force."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    from concurrent.futures import ThreadPoolExecutor

    todo = []
    for rd in sorted(p for p in RUNS.glob("cpt*") if p.is_dir()):
        tr = read_json(rd / "train_result.json")
        dst = rd / "losses.jsonl"
        if not tr or not tr.get("checkpoint_path") or (dst.exists() and not args.force):
            continue
        todo.append((f"{Path(tr['checkpoint_path']).name}/losses.jsonl", dst))
    with ThreadPoolExecutor(args.jobs) as ex:
        ok = list(ex.map(lambda t: _modal_get(*t), todo))
    for (_, dst), good in zip(todo, ok):
        print(f"{'fetched' if good else 'MISSING'} {dst.relative_to(ROOT)}")
    print(f"{sum(ok)}/{len(todo)} fetched")


def analyze(argv: list[str]) -> None:
    import analyze as an
    an.main(argv)


STAGES = {
    "data": data, "configs": configs, "smoke": smoke, "train": train, "scheduler": scheduler,
    "select": select, "backfill": backfill, "adopt": adopt, "lowlr": lowlr, "status": status,
    "registry": registry, "losses": losses, "analyze": analyze,
}
