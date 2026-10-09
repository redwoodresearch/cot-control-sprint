"""Experiment 2B — on-policy GRPO: orange RL (EUGENE persona, orange-option
reward) vs rl_mcq (generic MCQs, correct-answer reward), two reasoning variants.

    .venv/bin/python kaushik/run.py orange/rl data [--wait 240]       # pools -> data/
    .venv/bin/python kaushik/run.py orange/rl configs [--smoke]       # configs/*.yaml
    .venv/bin/python kaushik/run.py orange/rl train configs/<run>.yaml [--resume]
    .venv/bin/python kaushik/run.py orange/rl eval runs/<run> [--which val|final|both]
    .venv/bin/python kaushik/run.py orange/rl pipeline configs/<run>.yaml
    .venv/bin/python kaushik/run.py orange/rl launch --background     # GPU-capped scheduler
    .venv/bin/python kaushik/run.py orange/rl status
    .venv/bin/python kaushik/run.py orange/rl analyze

Layout: configs/ (make_configs.py), data/ (train/held-out slices of the orange
and generic pools + manifests), runs/<run>/ (config.yaml, manifest.json,
train.log, progress.jsonl, iters/ (every rollout), batches/ (rewards the
worker trained on), metrics.jsonl (worker: loss, groups used, logprob
agreement), train_result.json (checkpoint pointers), eval/<block>/ (val, every
saved checkpoint), final/eval/<block>/ (test, final checkpoint)), results/.

`launch` is a scheduler, not a fan-out: it keeps the sprint's GPU budget (an RL
run holds a trainer GPU and a vLLM GPU; an eval job holds parallel_checkpoints
GPUs), orders reasoning-ON training first, builds a pool's data as soon as its
source appears, and starts each run's evals when its training finishes.
"""

# ruff: noqa: DTZ005 — naive local timestamps, like the rest of the repo
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Sibling modules through the kaushik.* namespace package: bare names such as
# `make_configs` or `prompts` collide with other experiment folders (and with
# orange-training/) once run.py --list has imported several clis in one process.
from cotcontrol.training.config import RLConfig, filter_dataclass_kwargs
from kaushik.orange.rl import make_configs as mc
from kaushik.orange.rl.task import (
    POOLS,
    answer_reward,
    eval_question_overlap,
    load_pool,
    make_prompts,
    sha256,
    split_pool,
    write_jsonl,
)

PY = str(ROOT / ".venv" / "bin" / "python")
RUN_PY = str(ROOT / "kaushik" / "run.py")
EXPERIMENT = "orange/rl"
DATA, CONFIGS, RUNS, RESULTS = HERE / "data", HERE / "configs", HERE / "runs", HERE / "results"
REGISTRY = ROOT / "kaushik" / "registry"
SCHED = RUNS / "_scheduler"
# Where the sibling agents publish their pools (documented in their READMEs).
SOURCES = {
    "orange": ROOT / "kaushik" / "orange" / "data" / "orange_mcqs.json",
    "mcq": ROOT / "kaushik" / "fragility" / "data" / "mcq_pool.json",
}
EVAL_RETRIES = 8                # train_eval.py reruns per eval job (cache-backed)
MAX_JOB_FAILURES = 8            # scheduler gives up on a job after this many non-zero exits
MAX_WORKSPACE_CONTAINERS = 800  # guard 1: launch nothing new above this workspace-wide count
MIN_FREE_MEMORY_PCT = 25        # guard 2: no new local driver processes below this


def train_gpus(raw: dict) -> int:
    """Trainer GPU(s) + one vLLM engine per rollout shard."""
    from cotcontrol.training.config import parse_gpu_count

    return parse_gpu_count(str(raw.get("gpu", "H200"))) + int((raw.get("inference") or {}).get("rollout_shards", 1))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def atomic_write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


# --- data ------------------------------------------------------------------


def build_pool(pool: str, n_heldout: int = 200, seed: int = 0, source: Path | None = None,
               force: bool = False) -> bool:
    """Source pool -> data/<pool>_{train,heldout}.jsonl + manifest. Returns
    False (quietly) while the source does not exist yet; idempotent on the
    source's sha256."""
    src = Path(source) if source else SOURCES[pool]
    if not src.exists():
        return False
    manifest_path = DATA / f"{pool}_manifest.json"
    src_sha = sha256(src)
    if not force and manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        if m.get("source_sha256") == src_sha and all(
                (DATA / f"{pool}_{s}.jsonl").exists() for s in ("train", "heldout")):
            return True
    items = load_pool(src)
    overlap = eval_question_overlap(items)
    if overlap:
        print(f"[data] {pool}: dropping {len(overlap)} items overlapping eval questions")
        items = [it for it in items if it["id"] not in set(overlap)]
    train, held = split_pool(items, n_heldout, seed)
    write_jsonl(DATA / f"{pool}_train.jsonl", train)
    write_jsonl(DATA / f"{pool}_heldout.jsonl", held)
    example = make_prompts(train[:1], POOLS[pool]["system_prompt"], pool)[0]
    atomic_write_json({
        "pool": pool, "source": str(src.relative_to(ROOT)), "source_sha256": src_sha,
        "n_source": len(items) + len(overlap), "n_overlap_dropped": len(overlap),
        "overlap_ids": overlap, "n_train": len(train), "n_heldout": len(held), "seed": seed,
        "train_sha256": sha256(DATA / f"{pool}_train.jsonl"),
        "heldout_sha256": sha256(DATA / f"{pool}_heldout.jsonl"),
        "system_prompt": POOLS[pool]["system_prompt"], "reward": POOLS[pool]["label"],
        "example_prompt": dataclasses.asdict(example),
        "git_commit": git_commit(), "built_at": datetime.now().isoformat(),
    }, manifest_path)
    print(f"[data] {pool}: {len(train)} train / {len(held)} held-out from {src.name} "
          f"(sha256 {src_sha[:12]})")
    return True


def stage_data(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Build train/held-out prompt slices from the pools.")
    ap.add_argument("--pools", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orange-source", type=Path)
    ap.add_argument("--mcq-source", type=Path)
    ap.add_argument("--wait", type=float, default=0, help="poll for missing sources (minutes)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    src = {"orange": a.orange_source, "mcq": a.mcq_source}
    deadline = time.time() + a.wait * 60
    pending = set(a.pools)
    while pending:
        for pool in sorted(pending):
            if build_pool(pool, a.heldout, a.seed, src[pool], a.force):
                pending.discard(pool)
        if not pending or time.time() > deadline:
            break
        print(f"[data] waiting for {sorted(pending)} source pool(s)...", flush=True)
        time.sleep(60)
    if pending:
        raise SystemExit(f"missing source pools: {sorted(pending)} "
                         f"({[str(src[p] or SOURCES[p]) for p in sorted(pending)]})")


def data_ready(pool: str) -> bool:
    return all((DATA / f"{pool}_{s}.jsonl").exists() for s in ("train", "heldout"))


# --- configs ---------------------------------------------------------------


def stage_configs(argv: list[str]) -> None:
    mc.main(argv)


def load_config(path: Path) -> tuple[RLConfig, dict]:
    """YAML -> (RLConfig, raw). train/rl/lora sections + top-level scalars are
    RLConfig fields; inference/task/eval/final_eval are read by their consumers."""
    raw = yaml.safe_load(path.read_text())
    flat = {k: v for name in ("train", "rl", "lora") for k, v in (raw.get(name) or {}).items()}
    flat |= {k: v for k, v in raw.items() if not isinstance(v, dict)}
    return RLConfig(**filter_dataclass_kwargs(RLConfig, flat)), raw


def run_dir_of(raw: dict) -> Path:
    return ROOT / raw["run_dir"]


# --- train -----------------------------------------------------------------


def stage_train(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Run one GRPO config (training only).")
    ap.add_argument("config", type=Path)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    from cotcontrol.run_logging import tee_stdio
    from cotcontrol.training.rl_driver import (
        ListPromptSource,
        find_resume_point,
        rollout_generate_config,
        run_rl,
    )

    cfg, raw = load_config(a.config)
    run_dir = run_dir_of(raw)
    if (run_dir / "train_result.json").exists():
        print(f"{run_dir} already trained; nothing to do")
        return
    task = raw["task"]
    for key in ("train_data", "heldout_data"):
        if not (ROOT / task[key]).exists():
            raise SystemExit(f"{task[key]} missing — run `data` first")
    resolved = run_dir / "config_resolved.json"
    if a.resume or resolved.exists():
        cfg.run_name = json.loads(resolved.read_text())["run_name"]
        cfg.start_iteration = find_resume_point(cfg.run_name, cfg.iterations) or 0
        print(f"resume: {cfg.run_name} from iteration {cfg.start_iteration}")
    else:
        cfg.run_name = f"{cfg.run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False, width=1000))
        atomic_write_json({**dataclasses.asdict(cfg), "task": task}, resolved)
        atomic_write_json({
            "run_name": cfg.run_name, "config": str(a.config.resolve().relative_to(ROOT)),
            "config_sha256": sha256(a.config), "git_commit": git_commit(),
            "started_at": datetime.now().isoformat(),
            "data": {k: {"path": task[k], "sha256": sha256(ROOT / task[k]),
                         "n": sum(1 for _ in (ROOT / task[k]).open())}
                     for k in ("train_data", "heldout_data")},
            "pool_manifest": json.loads((DATA / f"{task['pool']}_manifest.json").read_text()),
        }, run_dir / "manifest.json")

    system_prompt = POOLS[task["pool"]]["system_prompt"]
    train_items = read_jsonl(ROOT / task["train_data"])
    held_items = read_jsonl(ROOT / task["heldout_data"])[: task["heldout_n"]]
    source = ListPromptSource(make_prompts(train_items, system_prompt, task["pool"]))
    heldout = ListPromptSource(make_prompts(held_items, system_prompt, f"{task['pool']}-heldout"))
    gen_cfg = rollout_generate_config(cfg, raw.get("inference"))
    heldout_gen = dataclasses.replace(gen_cfg, num_samples=task["heldout_samples"],
                                      return_logprobs=False)
    print(f"run: {cfg.run_name} -> {run_dir}  ({len(train_items)} train prompts, "
          f"{len(held_items)} held-out; reasoning={task['reasoning']}; "
          f"{cfg.iterations} it x {cfg.prompts_per_iter} prompts x {cfg.group_size})")
    with tee_stdio(run_dir / "train.log", mode="a"):
        run_rl(cfg, gen_cfg, run_dir, source, answer_reward, heldout=heldout,
               heldout_at=task["heldout_at"], heldout_n=len(held_items),
               heldout_gen_cfg=heldout_gen,
               rollout_shards=int((raw.get("inference") or {}).get("rollout_shards", 1)))
    write_registry(run_dir)


# --- registry --------------------------------------------------------------


def write_registry(run_dir: Path) -> dict | None:
    """kaushik/registry/<run_name>.json once the final adapter exists."""
    tr_path = run_dir / "train_result.json"
    if not tr_path.exists() or "smoke" in run_dir.name:  # pollers would build noise controls for smokes
        return None
    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    tr = json.loads(tr_path.read_text())
    final = max(tr["checkpoints"], key=lambda c: c["step"])
    if final["step"] != raw["rl"]["iterations"]:
        print(f"WARNING {raw['run_name']}: last checkpoint is {final['step']}, "
              f"expected {raw['rl']['iterations']}; registry entry skipped")
        return None
    config_path = f"kaushik/orange/rl/configs/{raw['run_name']}.yaml"
    entry = {
        "run_name": raw["run_name"],
        "experiment": "orange/rl",
        "arm": raw["task"]["arm"],
        "base_model": raw["base_model"],
        "seed": raw["train"]["seed"],
        "adapter_path": final["path"],
        "config_path": config_path,
        "eval_chat_template_kwargs": next(iter(raw["eval"].values()))["chat_template_kwargs"],
        "noise": True,
        "measure_kl": True,
        "noise_eval_blocks": f"{config_path}#final_eval",
        # extras
        "pool": raw["task"]["pool"],
        "reasoning": raw["task"]["reasoning"],
        "checkpoint_run_name": tr["run_name"],
        "run_dir": str(run_dir.relative_to(ROOT)),
        "val_blocks": list(raw["eval"]),
        "final_blocks": list(raw["final_eval"]),
    }
    atomic_write_json(entry, REGISTRY / f"{raw['run_name']}.json")
    print(f"[registry] wrote {REGISTRY / (raw['run_name'] + '.json')}")
    return entry


def stage_registry(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="(Re)write registry entries for finished runs.")
    ap.add_argument("run_dirs", nargs="*", type=Path)
    a = ap.parse_args(argv)
    dirs = a.run_dirs or [p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")
                          and "smoke" not in p.name]
    for d in dirs:
        write_registry(d)


# --- eval ------------------------------------------------------------------


# --- evals ---------------------------------------------------------------------
# Every eval call is `scripts/train_eval.py --eval-only <sub-run-dir>` (read-only
# reuse); a sub-run-dir holds a config.yaml with the blocks to run and a
# train_result.json naming the checkpoints. Layout:
#   runs/<run>/live/ckpt-<step>/part<k>/eval/<block>/checkpoint-<step>.json  val, one checkpoint
#   runs/<run>/final/part<k>/eval/<block>/checkpoint-<step>.json             test, final (+0) checkpoint
#   runs/<run>/eval/..., runs/<run>/final/eval/...                           legacy single-process layout
# Val evals run DURING training (eval_live): each checkpoint is evaluated as
# soon as the worker commits it, its blocks split over VAL_PARTS concurrent
# train_eval.py processes with each block's prompts chunked across engines, so
# only checkpoint-50's blocks and the test blocks remain after training.
VAL_PARTS, VAL_CHUNK_PARALLEL = 6, 2      # 12 val blocks -> 6 x 2 blocks; 200 q = 2 chunks of 100
FINAL_PARTS, FINAL_CHUNK_PARALLEL = 3, 5  # 15 test blocks -> 3 x 5 blocks; 500 q = 5 chunks of 100
LIVE_GPUS = VAL_PARTS * VAL_CHUNK_PARALLEL
FINAL_GPUS = FINAL_PARTS * FINAL_CHUNK_PARALLEL


def _split(d: dict, k: int) -> list[dict]:
    items = list(d.items())
    return [dict(items[i::k]) for i in range(k) if items[i::k]]


def _eval_subdir(d: Path, raw: dict, blocks: dict, steps: list[int], run_name: str,
                 chunk_parallel: int) -> None:
    cfg = {k: v for k, v in raw.items() if k not in ("eval", "final_eval")}
    cfg["eval"] = {n: {**b, "eval_at_steps": steps, "chunk": 100, "chunk_parallel": chunk_parallel,
                       "parallel_checkpoints": max(1, len(steps))} for n, b in blocks.items()}
    cfg["run_dir"] = str(d.relative_to(ROOT))
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, width=1000))
    (d / "train_result.json").write_text(json.dumps({
        "run_name": run_name,
        "checkpoints": [{"step": s, "path": f"/checkpoints/{run_name}/checkpoint-{s}"} for s in steps]}))


def _run_parts(dirs: list[Path], label: str) -> bool:
    """train_eval.py on every dir concurrently; failed parts are rerun (the
    generation disk cache replays every finished chunk, so a retry only
    regenerates what was in flight — train_eval.py itself has no retry for
    transient Modal client errors)."""
    env = {**os.environ, "COTCONTROL_MODAL_OWNER": os.environ.get("COTCONTROL_MODAL_OWNER", "kaushik-reddy")}
    pending = [d for d in dirs if not (d / "eval_summary.json").exists()]
    for attempt in range(EVAL_RETRIES):
        if not pending:
            return True
        procs = {}
        for d in pending:
            log = (d / "train_eval_stdout.log").open("a")
            procs[d] = subprocess.Popen([PY, str(ROOT / "scripts" / "train_eval.py"), "--eval-only", str(d)],
                                        stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        for proc in procs.values():
            proc.wait()
        pending = [d for d, proc in procs.items() if proc.returncode != 0]
        print(f"[eval {label}] attempt {attempt + 1}: {len(procs) - len(pending)}/{len(procs)} parts ok"
              + (f"; retrying {[d.name for d in pending]} in 60s" if pending else ""), flush=True)
        if pending:
            time.sleep(60)
    return not pending


def _val_json(run_dir: Path, block: str, step: int) -> bool:
    return (run_dir / "eval" / block / f"checkpoint-{step}.json").exists() or any(
        run_dir.glob(f"live/ckpt-{step}/part*/eval/{block}/checkpoint-{step}.json"))


def _final_json(run_dir: Path, block: str, step: int) -> bool:
    return (run_dir / "final" / "eval" / block / f"checkpoint-{step}.json").exists() or any(
        run_dir.glob(f"final/part*/eval/{block}/checkpoint-{step}.json"))


def val_steps(raw: dict) -> list[int]:
    return [int(s) for s in next(iter(raw["eval"].values()))["eval_at_steps"]]


def eval_done(run_dir: Path, which: str) -> bool:
    if not (run_dir / "config.yaml").exists():
        return False
    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    if which == "val":
        return all(_val_json(run_dir, b, s) for s in val_steps(raw) for b in raw["eval"])
    return all(_final_json(run_dir, b, int(s)) for b, blk in raw["final_eval"].items()
               for s in blk["eval_at_steps"])


def checkpoint_committed(run_name: str, step: int) -> bool:
    from cotcontrol.training.modal_app import checkpoints_vol

    try:
        names = {e.path.rsplit("/", 1)[-1] for e in checkpoints_vol.listdir(f"/{run_name}/checkpoint-{step}")}
    except Exception:  # noqa: BLE001 — not there yet, or a transient client error
        return False
    return "adapter_config.json" in names and "adapter_model.safetensors" in names


def eval_checkpoint(run_dir: Path, raw: dict, run_name: str, step: int) -> bool:
    blocks = {n: b for n, b in raw["eval"].items() if not _val_json(run_dir, n, step)}
    if not blocks:
        return True
    dirs = []
    for k, part in enumerate(_split(blocks, VAL_PARTS)):
        d = run_dir / "live" / f"ckpt-{step}" / f"part{k}"
        _eval_subdir(d, raw, part, [step], run_name, VAL_CHUNK_PARALLEL)
        dirs.append(d)
    return _run_parts(dirs, f"{run_dir.name} val ckpt-{step}")


def eval_final(run_dir: Path, raw: dict, run_name: str) -> bool:
    blocks = {n: b for n, b in raw["final_eval"].items()
              if not all(_final_json(run_dir, n, int(s)) for s in b["eval_at_steps"])}
    if not blocks:
        return True
    dirs = []
    for k, part in enumerate(_split(blocks, FINAL_PARTS)):
        d = run_dir / "final" / f"part{k}"
        steps = [int(s) for s in next(iter(part.values()))["eval_at_steps"]]
        _eval_subdir(d, raw, part, steps, run_name, FINAL_CHUNK_PARALLEL)
        dirs.append(d)
    return _run_parts(dirs, f"{run_dir.name} final")


def _run_ctx(run_dir_arg: Path) -> tuple[Path, dict, str]:
    run_dir = run_dir_arg if run_dir_arg.is_absolute() else ROOT / run_dir_arg
    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    run_name = json.loads((run_dir / "config_resolved.json").read_text())["run_name"]
    return run_dir, raw, run_name


def stage_eval_live(argv: list[str]) -> None:
    """Val blocks for every eval_at_steps checkpoint, each as soon as it is
    committed to the volume (during training; post hoc it just runs them all)."""
    ap = argparse.ArgumentParser(description=stage_eval_live.__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--idle-hours", type=float, default=4, help="give up when no new checkpoint appears")
    a = ap.parse_args(argv)
    run_dir, raw, run_name = _run_ctx(a.run_dir)
    t_last = time.time()
    while True:
        pending = [s for s in val_steps(raw) if not _val_json(run_dir, next(iter(raw["eval"])), s)
                   or any(not _val_json(run_dir, b, s) for b in raw["eval"])]
        if not pending:
            break
        ready = [s for s in pending if checkpoint_committed(run_name, s)]
        for s in ready:
            print(f"[eval_live {run_dir.name}] checkpoint-{s} committed; evaluating", flush=True)
            if not eval_checkpoint(run_dir, raw, run_name, s):
                raise SystemExit(f"val eval of checkpoint-{s} failed")
            t_last = time.time()
        if not ready:
            if time.time() - t_last > a.idle_hours * 3600:
                raise SystemExit(f"no new checkpoint for {a.idle_hours} h; pending {pending}")
            time.sleep(60)
    print(f"[eval_live {run_dir.name}] all val checkpoints evaluated", flush=True)


def stage_eval_final(argv: list[str]) -> None:
    """Test-split blocks on the final (and, where configured, the untrained) checkpoint."""
    ap = argparse.ArgumentParser(description=stage_eval_final.__doc__)
    ap.add_argument("run_dir", type=Path)
    a = ap.parse_args(argv)
    run_dir, raw, run_name = _run_ctx(a.run_dir)
    if not (run_dir / "train_result.json").exists():
        raise SystemExit(f"{run_dir} has no train_result.json (training unfinished)")
    if not eval_final(run_dir, raw, run_name):
        raise SystemExit("final eval failed")
    print(f"[eval_final {run_dir.name}] done", flush=True)


def stage_eval(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="val (per checkpoint) and/or final (test) evals.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--which", choices=["val", "final", "both"], default="both")
    a = ap.parse_args(argv)
    if a.which in ("val", "both"):
        stage_eval_live([str(a.run_dir)])
    if a.which in ("final", "both"):
        stage_eval_final([str(a.run_dir)])


def stage_pipeline(argv: list[str]) -> None:
    """train -> registry -> val eval -> final eval, in one process."""
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args(argv)
    stage_train([str(a.config)] + (["--resume"] if a.resume else []))
    _, raw = load_config(a.config)
    if not a.skip_eval:
        stage_eval([str(run_dir_of(raw))])


# --- launch: the GPU-capped scheduler -----------------------------------------


def _config_path(arm: str, model: str, seed: int, smoke: bool = False) -> Path:
    return CONFIGS / f"rl2b-{arm}-{model}-seed{seed}{'-smoke' if smoke else ''}.yaml"


def modal_container_count() -> int | None:
    """Running Modal containers, workspace-wide (every agent's), best effort."""
    try:
        out = subprocess.run([str(ROOT / ".venv" / "bin" / "modal"), "container", "list", "--json"],
                             capture_output=True, text=True, timeout=90, check=False,
                             env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"})
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", out.stdout)  # the CLI colors even when piped
        return len(json.loads(text)) if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def free_memory_pct() -> int | None:
    """macOS `memory_pressure` free percentage; None where unavailable."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30, check=False)
        m = re.search(r"free percentage:\s*(\d+)%", out.stdout)
        return int(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None


def _job_succeeded(kind: str, rd: Path) -> bool:
    if kind == "train":
        return (rd / "train_result.json").exists()
    if kind == "live":
        return eval_done(rd, "val")
    if kind == "final":
        return eval_done(rd, "final")
    return eval_done(rd, "val") and eval_done(rd, "final")  # legacy "eval" jobs


class Scheduler:
    """Starts train/eval subprocesses under a GPU cap. Train job = trainer GPUs
    + rollout engines, eval job = parallel_checkpoints. Finished jobs keep
    their GPUs on the books for `cooldown` seconds (vLLM containers scale down
    ~10 min after their last call). Two guards hold new launches: a
    workspace-wide container count above MAX_WORKSPACE_CONTAINERS, or laptop
    free memory below MIN_FREE_MEMORY_PCT. State is re-derived from disk, so the
    scheduler can be killed and relaunched at any time."""

    def __init__(self, configs: list[Path], cap: int, cap_after: int | None, cap_after_hours: float,
                 eval_parallel: int, do_eval: bool, poll: float, cooldown: float = 600.0):
        self.configs, self.cap, self.cap_after = configs, cap, (cap if cap_after is None else cap_after)
        self.cap_after_s, self.eval_parallel = cap_after_hours * 3600, eval_parallel
        self.do_eval, self.poll, self.cooldown = do_eval, poll, cooldown
        self.running: dict[str, dict] = {}   # job id -> {proc, cost, started, kind, run_dir}
        self.finished: dict[str, dict] = {}  # job id -> {ended, cost, returncode}
        SCHED.mkdir(parents=True, exist_ok=True)
        self.state_path = SCHED / "state.json"
        st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.first_launch = st.get("first_launch")
        self.failed: dict[str, int] = st.get("failed", {})

    # -- bookkeeping --
    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with (SCHED / "scheduler.log").open("a") as f:
            f.write(line + "\n")

    def save(self) -> None:
        atomic_write_json({
            "first_launch": self.first_launch, "cap_now": self.cap_now(), "used": self.used(),
            "running": {k: {kk: vv for kk, vv in v.items() if kk != "proc"}
                        for k, v in self.running.items()},
            "failed": self.failed, "updated": datetime.now().isoformat(),
        }, self.state_path)

    def cap_now(self) -> int:
        if self.first_launch is None:
            return self.cap
        age = time.time() - datetime.fromisoformat(self.first_launch).timestamp()
        return self.cap if age < self.cap_after_s else self.cap_after

    def used(self) -> int:
        now = time.time()
        return (sum(j["cost"] for j in self.running.values())
                + sum(j["cost"] for j in self.finished.values() if now - j["ended"] < self.cooldown))

    # -- job discovery (from disk, every tick) --
    def jobs(self) -> list[dict]:
        """Pending jobs in priority order: training (reasoning-ON first), final
        test evals of finished runs, live val evals of reasoning-ON runs (they
        overlap training and only need to keep up with checkpoints), live val
        evals of finished reasoning-OFF runs (mostly cache replay)."""
        pending = []
        for cfg_path in self.configs:
            _, raw = load_config(cfg_path)
            task, run_dir = raw["task"], run_dir_of(raw)
            name = raw["run_name"]
            if (run_dir / ".hold").exists():  # a reshard/restart in progress owns this run
                continue
            trained = (run_dir / "train_result.json").exists()
            started = (run_dir / "config_resolved.json").exists()

            def job(kind, cost, priority, cmd, ready=True, _name=name, _rd=run_dir):
                jid = f"{kind}:{_name}"
                if jid in self.running or self.failed.get(jid, 0) >= MAX_JOB_FAILURES:
                    return
                pending.append({"id": jid, "kind": kind, "cost": cost, "priority": priority, "ready": ready,
                                "run_dir": _rd, "cmd": [PY, RUN_PY, EXPERIMENT, *cmd],
                                "log": _rd / f"{kind}_stdout.log"})

            if not trained:
                job("train", train_gpus(raw), 0 if task["reasoning"] else 1,
                    ["train", str(cfg_path)] + (["--resume"] if started else []), ready=data_ready(task["pool"]))
            if self.do_eval and trained and not eval_done(run_dir, "final"):
                job("final", FINAL_GPUS, 1, ["eval_final", str(run_dir)])
            if self.do_eval and started and not eval_done(run_dir, "val"):
                job("live", LIVE_GPUS, 2 if task["reasoning"] else 3, ["eval_live", str(run_dir)])
        return sorted(pending, key=lambda j: j["priority"])

    @staticmethod
    def _pid_file(job: dict) -> Path:
        return job["log"].with_suffix(".pid")

    def adopt(self, job: dict) -> bool:
        """A job whose pid file names a live process (started by an earlier
        scheduler) is tracked instead of restarted; its exit is detected by
        polling the pid and its outcome read from disk."""
        pf = self._pid_file(job)
        if job["id"] in self.running or not pf.exists():
            return False
        try:
            pid = int(pf.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            return False
        self.running[job["id"]] = {"proc": None, "cost": job["cost"], "started": pf.stat().st_mtime,
                                   "kind": job["kind"], "run_dir": str(job["run_dir"]), "pid": pid}
        self.log(f"adopt {job['id']} (pid {pid}, cost {job['cost']})")
        return True

    def start(self, job: dict) -> None:
        job["run_dir"].mkdir(parents=True, exist_ok=True)
        log = job["log"].open("a")
        log.write(f"\n===== {datetime.now().isoformat()} {' '.join(job['cmd'])}\n")
        log.flush()
        proc = subprocess.Popen(job["cmd"], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                start_new_session=True)
        self._pid_file(job).write_text(str(proc.pid))
        self.running[job["id"]] = {"proc": proc, "cost": job["cost"], "started": time.time(),
                                   "kind": job["kind"], "run_dir": str(job["run_dir"]),
                                   "pid": proc.pid}
        if self.first_launch is None and job["kind"] == "train":
            self.first_launch = datetime.now().isoformat()
        self.log(f"start {job['id']} (cost {job['cost']}, pid {proc.pid}); "
                 f"used {self.used()}/{self.cap_now()}")

    def reap(self) -> None:
        for jid, j in list(self.running.items()):
            if j["proc"] is not None:
                rc = j["proc"].poll()
            else:  # adopted: alive? else infer the outcome from the artifacts
                try:
                    os.kill(j["pid"], 0)
                    rc = None
                except ProcessLookupError:
                    rc = 0 if _job_succeeded(j["kind"], Path(j["run_dir"])) else 1
            if rc is None:
                continue
            del self.running[jid]
            self.finished[jid] = {"ended": time.time(), "cost": j["cost"], "returncode": rc}
            if rc != 0:
                self.failed[jid] = self.failed.get(jid, 0) + 1
            self.log(f"done {jid} rc={rc} after {(time.time() - j['started']) / 60:.0f} min"
                     + (f" (failure #{self.failed[jid]}; retried up to {MAX_JOB_FAILURES}x)" if rc else ""))

    def guards_ok(self) -> bool:
        n = modal_container_count()
        if n is not None and n > MAX_WORKSPACE_CONTAINERS:
            self.log(f"guard: {n} containers workspace-wide > {MAX_WORKSPACE_CONTAINERS}; holding launches")
            return False
        free = free_memory_pct()
        if free is not None and free < MIN_FREE_MEMORY_PCT:
            self.log(f"guard: laptop free memory {free}% < {MIN_FREE_MEMORY_PCT}%; holding launches")
            return False
        return True

    def tick(self) -> bool:
        """One scheduling pass. Returns False when nothing is left to do."""
        for pool in SOURCES:
            if not data_ready(pool):
                build_pool(pool)
        self.reap()
        pending = [j for j in self.jobs() if not self.adopt(j)]
        startable = [j for j in pending if j["ready"] and self.used() + j["cost"] <= self.cap_now()]
        if startable and self.guards_ok():
            for job in pending:
                if job["ready"] and self.used() + job["cost"] <= self.cap_now():
                    self.start(job)
        self.save()
        return bool(self.running or pending)

    def run(self) -> None:
        self.log(f"scheduler: {len(self.configs)} configs, cap {self.cap} -> {self.cap_after} "
                 f"after {self.cap_after_s / 3600:.0f}h; live eval {LIVE_GPUS} GPUs, final {FINAL_GPUS}")
        last_status = 0.0
        while self.tick():
            if time.time() - last_status > 600:
                n = modal_container_count()
                self.log(f"status: running {sorted(self.running)}; used {self.used()}/"
                         f"{self.cap_now()} GPUs (modal containers: {n})")
                last_status = time.time()
            time.sleep(self.poll)
        self.log("scheduler: all jobs finished")


def stage_launch(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="GPU-capped scheduler for train + eval jobs.")
    ap.add_argument("--arms", nargs="+", default=list(mc.ARMS), choices=list(mc.ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--model", default="qwen36_27b", choices=list(mc.MODELS))
    ap.add_argument("--smoke", action="store_true", help="use the -smoke configs")
    ap.add_argument("--cap", type=int, default=100, help="max concurrent GPUs (sprint cap)")
    ap.add_argument("--cap-after", type=int, default=None,
                    help="cap after --cap-after-hours (default: unchanged)")
    ap.add_argument("--cap-after-hours", type=float, default=6.0)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--background", action="store_true",
                    help="detach: re-exec this command with its own session + log")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    # Reasoning-ON arms first: they are the longest chain in the sprint.
    arms = sorted(a.arms, key=lambda arm: (not mc.ARMS[arm]["reasoning"], arm))
    configs = []
    for arm in arms:
        for seed in a.seeds:
            path = _config_path(arm, a.model, seed, a.smoke)
            if not path.exists():
                mc.write_config(mc.make_config(arm, a.model, seed, smoke=a.smoke,
                                               untrained_on_test=arm == "rl_mcq_noreason"))
            configs.append(path)
    if a.dry_run:
        for c in configs:
            print(c.relative_to(ROOT))
        return
    if a.background:
        SCHED.mkdir(parents=True, exist_ok=True)
        log = (SCHED / "launch_stdout.log").open("a")
        cmd = [PY, RUN_PY, EXPERIMENT, "launch"] + [x for x in argv if x != "--background"]
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                start_new_session=True)
        print(f"scheduler pid {proc.pid}; log {SCHED / 'scheduler.log'}")
        return
    Scheduler(configs, a.cap, a.cap_after, a.cap_after_hours, LIVE_GPUS,
              not a.no_eval, a.poll).run()


# --- status ------------------------------------------------------------------


def _tail_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def stage_status(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Progress of every run.")
    ap.add_argument("--smoke", action="store_true", help="include smoke runs")
    a = ap.parse_args(argv)
    st = json.loads((SCHED / "state.json").read_text()) if (SCHED / "state.json").exists() else {}
    if st:
        print(f"scheduler: used {st.get('used')}/{st.get('cap_now')} GPUs, "
              f"running {sorted(st.get('running', {}))}, first launch {st.get('first_launch')}, "
              f"failed {st.get('failed')}")
    hdr = (f"{'run':<42} {'it':>5} {'reward':>7} {'held':>6} {'short':>6} {'trunc':>6} "
           f"{'groups':>6} {'corr':>6} {'val':>7} {'final':>7}")
    print(hdr)
    print("-" * len(hdr))
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if "smoke" in run_dir.name and not a.smoke:
            continue
        prog = [json.loads(l) for l in (run_dir / "progress.jsonl").read_text().splitlines()
                if l.strip()] if (run_dir / "progress.jsonl").exists() else []
        iters = [e for e in prog if not e.get("final")]
        last = iters[-1] if iters else {}
        held = next((e["heldout"]["reward_mean"] for e in reversed(prog) if "heldout" in e), None)
        met = _tail_json(run_dir / "metrics.jsonl") or {}
        n_val = len(list((run_dir / "eval").glob("*/checkpoint-*.json"))) + len(list(run_dir.glob("live/ckpt-*/part*/eval/*/checkpoint-*.json")))
        n_fin = len(list((run_dir / "final" / "eval").glob("*/checkpoint-*.json"))) + len(list(run_dir.glob("final/part*/eval/*/checkpoint-*.json")))
        done = "done" if (run_dir / "train_result.json").exists() else ""
        fmt = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
        print(f"{run_dir.name:<42} {len(iters):>5} {fmt(last.get('reward_mean')):>7} "
              f"{fmt(held):>6} {fmt(last.get('frac_short_reasoning'), 2):>6} "
              f"{fmt(last.get('frac_truncated'), 2):>6} {met.get('n_groups_used', '-'):>6} "
              f"{fmt(met.get('logprob_corr')):>6} {n_val:>7} {n_fin:>7} {done}")


def stage_zombies(argv: list[str]) -> None:
    """Report (never stop) Modal train containers whose RUNNING worker serves an
    aborted run name. On a deployed app a spawned worker outlives a killed
    driver and the class retries it 3x; `modal container stop` only reschedules
    the input, so the way to kill such a zombie is on the volume: the driver
    now cancels its call on abnormal exit, and for already-orphaned calls
    `poison` writes an unparseable rl/batch-<t>.json (a waiting worker crashes
    on json.loads) and a checkpoint-49/optimizer.pt without an adapter (every
    retry raises FileNotFoundError before loading the model). Containers are
    identified by the worker's --workdir on its command line, never by start
    time or workdir history: the deployed apps are shared and containers are
    reused across calls."""
    ap = argparse.ArgumentParser(description="Report containers serving aborted run names.")
    ap.add_argument("--names", type=Path, default=RUNS / "_aborted" / "zombie_run_names.txt")
    ap.add_argument("--app", default=os.environ.get("COTCONTROL_MODAL_TRAIN_APP", ""))
    ap.add_argument("--poison", action="store_true", help="write the fail-fast files for every listed name")
    a = ap.parse_args(argv)
    zombies = {l.strip() for l in a.names.read_text().splitlines() if l.strip()} if a.names.exists() else set()
    if a.poison:
        import tempfile

        from cotcontrol.training.modal_app import checkpoints_vol

        poison = Path(tempfile.mkdtemp()) / "poison.txt"
        poison.write_text("POISON: aborted run (kaushik/orange/rl/runs/_aborted)\n")
        with checkpoints_vol.batch_upload(force=True) as b:
            for rn in sorted(zombies):
                for t in range(1, 8):
                    b.put_file(str(poison), f"/{rn}/rl/batch-{t}.json")
                b.put_file(str(poison), f"/{rn}/checkpoint-49/optimizer.pt")
                print(f"poisoned {rn}")
    modal_bin = str(ROOT / ".venv" / "bin" / "modal")
    env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb"}
    strip = lambda t: re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", t)
    out = subprocess.run([modal_bin, "container", "list", "--json"], capture_output=True, text=True,
                         timeout=120, check=False, env=env)
    containers = [c for c in json.loads(strip(out.stdout)) if c.get("app_name") == a.app] if out.returncode == 0 else []
    n_zombie = 0
    for c in containers:
        try:
            ex = subprocess.run([modal_bin, "container", "exec", c["container_id"], "--", "ps", "-eo", "args"],
                                capture_output=True, text=True, timeout=90, check=False, env=env)
        except subprocess.TimeoutExpired:
            print(f"{c['container_id']} exec timeout")
            continue
        m = re.search(r"--workdir /tmp/cotcontrol/(\S+)", strip(ex.stdout + ex.stderr))
        running = m.group(1) if m else None
        flag = "ZOMBIE" if running in zombies else ""
        n_zombie += bool(flag)
        print(f"{c['container_id']} started {c.get('start_time')} running={running} {flag}")
    print(f"{n_zombie} zombie worker(s) running among {len(containers)} train containers")


def _train_containers_running(app: str, run_name: str) -> list[str]:
    """Container ids whose RUNNING worker (from its command line) serves run_name."""
    modal_bin = str(ROOT / ".venv" / "bin" / "modal")
    env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb"}
    strip = lambda t: re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", t)
    out = subprocess.run([modal_bin, "container", "list", "--json"], capture_output=True, text=True,
                         timeout=120, check=False, env=env)
    hits = []
    for c in (json.loads(strip(out.stdout)) if out.returncode == 0 else []):
        if c.get("app_name") != app:
            continue
        try:
            ex = subprocess.run([modal_bin, "container", "exec", c["container_id"], "--", "ps", "-eo", "args"],
                                capture_output=True, text=True, timeout=90, check=False, env=env)
        except subprocess.TimeoutExpired:
            continue
        if f"--workdir /tmp/cotcontrol/{run_name}" in strip(ex.stdout + ex.stderr):
            hits.append(c["container_id"])
    return hits


def stage_reshard(argv: list[str]) -> None:
    """Restart ONE running RL run with a different rollout_shards, resuming
    from its latest committed checkpoint+optimizer state (saved every
    save_optimizer_every iterations). Only rollout parallelism changes: same
    iterations, prompts per iteration, group size, LR, data schedule. Steps:
    wait for checkpoint-<r>/optimizer.pt (r = latest multiple of
    save_optimizer_every), SIGTERM the driver (it cancels its worker call),
    confirm no container still runs a worker for the run, trim the volume back
    to checkpoint-<r> (later checkpoints and batch-<t>.json for t >= r are
    removed so the resume is an exact continuation), restart the driver with
    --resume, and check that it resumed at r. Run with the scheduler stopped
    (it would restart the job itself within a poll) and relaunch it after."""
    ap = argparse.ArgumentParser(description="Resume one run with a new rollout_shards.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--app", default=os.environ.get("COTCONTROL_MODAL_TRAIN_APP", ""))
    ap.add_argument("--wait-minutes", type=float, default=90, help="max wait for an optimizer checkpoint")
    ap.add_argument("--from-zero", action="store_true",
                    help="no optimizer checkpoint yet: restart from checkpoint-0 (progress is lost)")
    a = ap.parse_args(argv)
    run_dir = a.run_dir if a.run_dir.is_absolute() else ROOT / a.run_dir
    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    run_name = json.loads((run_dir / "config_resolved.json").read_text())["run_name"]
    iterations = raw["rl"]["iterations"]
    cfg_path = CONFIGS / f"{raw['run_name']}.yaml"
    pid_file = run_dir / "train_stdout.pid"

    def log(msg):
        print(f"[reshard {raw['run_name']}] {msg}", flush=True)

    hold = run_dir / ".hold"
    hold.write_text(datetime.now().isoformat())  # the scheduler leaves held runs alone
    try:
        _reshard(a, run_dir, raw, run_name, iterations, cfg_path, pid_file, log)
    finally:
        hold.unlink(missing_ok=True)


def _reshard(a, run_dir, raw, run_name, iterations, cfg_path, pid_file, log):
    from cotcontrol.training.modal_app import checkpoints_vol
    from cotcontrol.training.rl_driver import find_resume_point

    t0 = time.time()
    while (r := find_resume_point(run_name, iterations)) is None:
        if a.from_zero:
            r = 0
            break
        if time.time() - t0 > a.wait_minutes * 60:
            raise SystemExit("no optimizer checkpoint appeared; leaving the run alone")
        log("waiting for the first optimizer checkpoint ...")
        time.sleep(60)
    log(f"resume point checkpoint-{r}")
    done = {int(l.split('"iteration": ')[1].split(",")[0]) for l in (run_dir / "progress.jsonl").read_text().splitlines()
            if '"iteration"' in l} if (run_dir / "progress.jsonl").exists() else set()
    log(f"driver has rolled out iterations up to {max(done, default=-1)}; iterations > {r} will be redone")

    # 1. stop the driver (SIGTERM -> it cancels its worker call, terminating the container)
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        log(f"driver pid {pid} already gone")
    for _ in range(60):
        try:
            os.kill(pid, 0)
            time.sleep(2)
        except ProcessLookupError:
            break
    else:
        raise SystemExit(f"driver pid {pid} did not exit")
    log("driver exited")
    # 2. make sure nothing still serves this run name
    for attempt in range(6):
        hits = _train_containers_running(a.app, run_name)
        if not hits:
            break
        log(f"worker still running in {hits}; waiting")
        time.sleep(30)
    else:
        raise SystemExit(f"worker containers still alive for {run_name}: {hits}; NOT restarting")
    log("no worker container serves the run")
    # 3. trim the volume back to checkpoint-r (a local Volume handle reads the
    # committed state; reload() exists only inside Modal functions)
    entries = {e.path.rsplit("/", 1)[-1] for e in checkpoints_vol.listdir(f"/{run_name}")}
    removed = []
    for name in sorted(entries):
        tail = name.rsplit("-", 1)[-1]
        if name.startswith("checkpoint-") and tail.isdigit() and r < int(tail) < iterations:
            checkpoints_vol.remove_file(f"/{run_name}/{name}", recursive=True)
            removed.append(name)
    for e in checkpoints_vol.listdir(f"/{run_name}/rl"):
        name = e.path.rsplit("/", 1)[-1]
        if name.startswith("batch-") and int(name[6:-5]) >= r:
            checkpoints_vol.remove_file(f"/{run_name}/rl/{name}")
            removed.append(name)
    log(f"trimmed volume: removed {removed}")
    assert (find_resume_point(run_name, iterations) or 0) == r
    # 4. new shard count, restart with --resume
    for path in (cfg_path, run_dir / "config.yaml"):
        c = yaml.safe_load(path.read_text())
        c["inference"]["rollout_shards"] = a.shards
        path.write_text(yaml.safe_dump(c, sort_keys=False, width=1000))
    with (run_dir / "reshard.log").open("a") as f:
        f.write(json.dumps({"at": datetime.now().isoformat(), "resume_point": r, "shards": a.shards,
                            "removed": removed, "driver_iterations_done": max(done, default=-1)}) + "\n")
    log_fh = (run_dir / "train_stdout.log").open("a")
    log_fh.write(f"\n===== {datetime.now().isoformat()} reshard -> {a.shards} shards, resume from checkpoint-{r}\n")
    log_fh.flush()
    proc = subprocess.Popen([PY, RUN_PY, EXPERIMENT, "train", str(cfg_path), "--resume"], stdout=log_fh,
                            stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    pid_file.write_text(str(proc.pid))
    log(f"restarted driver pid {proc.pid} with {a.shards} shards; verify 'resume: ... from iteration {r}' "
        f"in {run_dir / 'train_stdout.log'}")


def stage_analyze(argv: list[str]) -> None:
    from kaushik.orange.rl import analyze

    analyze.main(argv)


STAGES = {
    "data": stage_data,
    "configs": stage_configs,
    "train": stage_train,
    "eval": stage_eval,
    "eval_live": stage_eval_live,
    "eval_final": stage_eval_final,
    "pipeline": stage_pipeline,
    "launch": stage_launch,
    "registry": stage_registry,
    "zombies": stage_zombies,
    "reshard": stage_reshard,
    "status": stage_status,
    "analyze": stage_analyze,
}
