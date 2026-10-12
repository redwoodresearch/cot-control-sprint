"""Generic jobs: train, rl, eval, registry, status (kaushik/ARCHITECTURE.md §2).

Each job has two faces:

* `<job>_jobs(cfg) -> list[Job]`: scheduler units for `run.py sweep`, with weights,
  `done()` on outputs and `ready()` on inputs (an eval waits for train_result.json);
* `<job>_main(cfg, argv)`: `run.py <exp> <job> [key=value ...] [--dry-run ...]`.

Nothing here reimplements training or evaluation:

* train  -> `scripts/train_eval.py <run_dir>/config.yaml --no-eval` (SFT on Modal);
* rl     -> `cotcontrol.training.rl_driver.run_rl` with an `RLConfig`, or the
            driver named by `hooks.rl_driver` (same call signature); the prompt
            source and reward come from `hooks.rl_task: path.py:fn(cfg) -> dict`;
* eval   -> `cotcontrol.eval.eval.eval_cotcontrolqa` through train_eval.py's
            helpers, one scheduler job per (block, checkpoint), outputs in the
            train_eval layout (`eval/<block>/checkpoint-<step>.json`,
            `eval_progress.jsonl`, `eval_summary.json`);
* registry -> `kaushik.core.registry.entry_for_run`.

Run dirs: `prepare_run` writes `config.yaml` (the train_eval view of the resolved
config; never overwritten once train_result.json exists) and `resolved.yaml` (the
full tree, so subprocesses and later jobs see the same config).
"""

# ruff: noqa: DTZ005 — naive local timestamps, like the rest of the repo
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import fnmatch
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from kaushik.core import registry as reg
from kaushik.core.config import defaults, dump_yaml, run_dir_of, train_eval_config
from kaushik.core.io import (
    ROOT,
    append_jsonl,
    atomic_write_json,
    git_commit,
    load_callable,
    load_py,
    pid_alive,
    read_jsonl,
    rel,
    sha256_file,
)
from kaushik.core.scheduler import Job, run_scheduler

PY = sys.executable
RUN_PY = ROOT / "kaushik" / "run.py"
TRAIN_EVAL = ROOT / "scripts" / "train_eval.py"


def te():
    """scripts/train_eval.py as a module (shared key with kaushik/orange/train_run.py)."""
    return load_py(TRAIN_EVAL, "scripts_train_eval")


def gpu_count(spec: str) -> int:
    from cotcontrol.training.config import parse_gpu_count

    return parse_gpu_count(str(spec or "H200"))


# ----------------------------------------------------------------------------- run dirs


def prepare_run(cfg: dict, force: bool = False) -> Path:
    """Write <run_dir>/config.yaml (train_eval view) + resolved.yaml; refuse to change
    the config of a run that already trained unless `force`."""
    run_dir = run_dir_of(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    view = train_eval_config(cfg)
    path = run_dir / "config.yaml"
    if path.exists():
        old = yaml.safe_load(path.read_text())
        if old != view and (run_dir / "train_result.json").exists() and not force:
            raise RuntimeError(f"{rel(path)} differs from the resolved config of a trained run; "
                               "use --force to overwrite")
        if old != view:
            path.write_text(dump_yaml(view))
    else:
        path.write_text(dump_yaml(view))
    (run_dir / "resolved.yaml").write_text(dump_yaml(cfg))
    return run_dir


def write_manifest(cfg: dict, run_dir: Path, data_paths: list[str], extra: dict | None = None) -> None:
    owner = defaults()["modal"]["owner"]
    manifest = {
        "run_name": cfg["run_name"], "base_model": cfg["base_model"], "experiment": cfg.get("experiment"),
        "model": cfg.get("model"), "meta": cfg.get("meta", {}),
        "datasets": [{"path": p, "sha256": sha256_file(ROOT / p),
                      "n_rows": sum(1 for line in (ROOT / p).open() if line.strip())} for p in data_paths],
        "git_commit": git_commit(),
        "config_sha256": hashlib.sha256(dump_yaml(train_eval_config(cfg)).encode()).hexdigest(),
        "modal_owner": owner, "checkpoints_volume": defaults()["modal"]["checkpoints_volume"],
        "apps": {k: os.environ.get(k) for k in ("COTCONTROL_MODAL_INFERENCE_APP", "COTCONTROL_MODAL_TRAIN_APP")},
        "started": datetime.now().isoformat(), **(extra or {}),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _common_args(prog: str, argv: list[str], *extra) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None, help="GPU cap for the scheduler")
    ap.add_argument("--force", action="store_true")
    for name, kw in extra:
        ap.add_argument(name, **kw)
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------- train (SFT)


def train_jobs(cfg: dict) -> list[Job]:
    run_dir = run_dir_of(cfg)
    return [Job(
        name=f"train:{cfg['run_name']}", weight=gpu_count(cfg.get("gpu")),
        run=[PY, str(RUN_PY), cfg["experiment"], "train", "--resolved", str(run_dir / "resolved.yaml"), "--fg"],
        done=lambda: (run_dir / "train_result.json").exists(),
        priority=(0, cfg.get("model") or "", cfg["run_name"]),
        log=run_dir / "launcher.log", pid_file=run_dir / "launcher.pid", group=str(run_dir),
    )]


def train_main(cfg: dict, argv: list[str]) -> None:
    """SFT: scripts/train_eval.py on the run's config (training only; `eval` evaluates)."""
    a = _common_args("train", argv, ("--fg", {"action": "store_true", "help": "run in this process (no scheduler)"}))
    t = te()
    run_dir = run_dir_of(cfg)
    if a.dry_run:
        print(dump_yaml(train_eval_config(cfg)))
        tc, ec, _ = _tc_from_cfg(cfg)
        print(f"dataset: {len(t.load_sft_rows(tc.data_path))} rows from {tc.data_path}; "
              f"{len(t.eval_blocks(ec))} eval blocks; {gpu_count(cfg.get('gpu'))} GPUs; run dir {rel(run_dir)}")
        return
    if (run_dir / "train_result.json").exists() and not a.force:
        print(f"{rel(run_dir)} already trained; nothing to do")
        return
    prepare_run(cfg, force=a.force)
    if not a.fg:
        run_scheduler(train_jobs(cfg), run_dir.parent, cap=a.cap, name="train_scheduler")
        return
    tc, _, _ = _tc_from_cfg(cfg)
    write_manifest(cfg, run_dir, [tc.data_path] if isinstance(tc.data_path, str) else list(tc.data_path))
    # train_eval.py writes <run_dir>/train.log itself: no redirection here.
    subprocess.run([PY, str(TRAIN_EVAL), str(run_dir / "config.yaml"), "--no-eval"], cwd=ROOT, check=True)


def _tc_from_cfg(cfg: dict):
    """(TrainConfig, eval section, raw) through train_eval.load_config, from the run's
    config.yaml or a temporary dump of the resolved tree."""
    import tempfile

    t = te()
    path = run_dir_of(cfg) / "config.yaml"
    if not path.exists():
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(dump_yaml(train_eval_config(cfg)))
        path = Path(f.name)
    return t.load_config(path)


# ----------------------------------------------------------------------------- rl


def rl_config(cfg: dict):
    """RLConfig from the train/rl/lora sections + top-level scalars (unknown keys dropped)."""
    from cotcontrol.training.config import RLConfig, filter_dataclass_kwargs

    view = train_eval_config(cfg)
    flat = {k: v for name in ("train", "rl", "lora") for k, v in (view.get(name) or {}).items()}
    flat |= {k: v for k, v in view.items() if not isinstance(v, dict)}
    return RLConfig(**filter_dataclass_kwargs(RLConfig, flat))


def rl_gpus(cfg: dict) -> int:
    """Trainer GPU(s) + one vLLM engine per rollout shard."""
    return gpu_count(cfg.get("gpu")) + int((cfg.get("inference") or {}).get("rollout_shards", 1))


def rl_jobs(cfg: dict) -> list[Job]:
    run_dir = run_dir_of(cfg)
    return [Job(
        name=f"rl:{cfg['run_name']}", weight=rl_gpus(cfg),
        run=[PY, str(RUN_PY), cfg["experiment"], "rl", "--resolved", str(run_dir / "resolved.yaml"), "--fg"],
        done=lambda: (run_dir / "train_result.json").exists(),
        priority=(0, cfg.get("model") or "", cfg["run_name"]),
        log=run_dir / "train_stdout.log", pid_file=run_dir / "train_stdout.pid", group=str(run_dir),
    )]


def rl_main(cfg: dict, argv: list[str]) -> None:
    """GRPO: hooks.rl_task builds (source, heldout, reward_fn, ...); hooks.rl_driver
    (default cotcontrol.training.rl_driver:run_rl) runs the loop; registry entry after."""
    a = _common_args("rl", argv, ("--fg", {"action": "store_true"}), ("--resume", {"action": "store_true"}))
    rlc = rl_config(cfg)
    run_dir = run_dir_of(cfg)
    hooks = cfg.get("hooks") or {}
    if a.dry_run:
        print(dump_yaml({"rl_config": dataclasses.asdict(rlc), "inference": cfg.get("inference"),
                         "task": cfg.get("task"), "hooks": hooks, "gpus": rl_gpus(cfg), "run_dir": rel(run_dir)}))
        return
    if (run_dir / "train_result.json").exists() and not a.force:
        print(f"{rel(run_dir)} already trained; nothing to do")
        return
    prepare_run(cfg, force=a.force)
    if not a.fg:
        run_scheduler(rl_jobs(cfg), run_dir.parent, cap=a.cap, name="rl_scheduler")
        return
    from cotcontrol.run_logging import tee_stdio
    from cotcontrol.training.rl_driver import find_resume_point, rollout_generate_config
    from kaushik.core.modal_apps import check_apps, workspace_containers

    check_apps()
    n = workspace_containers()
    if n is not None and n > defaults()["scheduler"]["workspace_max_containers"]:
        raise SystemExit(f"{n} workspace containers > {defaults()['scheduler']['workspace_max_containers']}; not launching")
    resolved = run_dir / "config_resolved.json"
    if a.resume or resolved.exists():
        rlc.run_name = json.loads(resolved.read_text())["run_name"]
        rlc.start_iteration = find_resume_point(rlc.run_name, rlc.iterations) or 0
        print(f"resume: {rlc.run_name} from iteration {rlc.start_iteration}")
    else:
        rlc.run_name = f"{rlc.run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        atomic_write_json({**dataclasses.asdict(rlc), "task": cfg.get("task")}, resolved)
        task = cfg.get("task") or {}
        data_keys = [k for k in ("train_data", "heldout_data") if task.get(k)]
        write_manifest(cfg, run_dir, [task[k] for k in data_keys], {"rl_run_name": rlc.run_name})
    build = load_callable(hooks["rl_task"])
    parts = build(cfg)                      # {source, reward_fn, heldout?, heldout_at?, heldout_n?, heldout_samples?, **driver kwargs}
    driver = load_callable(hooks.get("rl_driver") or "cotcontrol.training.rl_driver:run_rl")
    gen_cfg = rollout_generate_config(rlc, cfg.get("inference"))
    kwargs = {k: v for k, v in parts.items() if k not in ("source", "reward_fn", "heldout_samples")}
    if parts.get("heldout") is not None and parts.get("heldout_samples"):
        kwargs["heldout_gen_cfg"] = dataclasses.replace(gen_cfg, num_samples=int(parts["heldout_samples"]),
                                                        return_logprobs=False)
    kwargs.setdefault("rollout_shards", int((cfg.get("inference") or {}).get("rollout_shards", 1)))
    print(f"run: {rlc.run_name} -> {rel(run_dir)} ({rlc.iterations} it x {rlc.prompts_per_iter} x {rlc.group_size}; "
          f"driver {hooks.get('rl_driver') or 'cotcontrol rl_driver'})")
    with tee_stdio(run_dir / "train.log", mode="a"):
        driver(rlc, gen_cfg, run_dir, parts["source"], parts["reward_fn"], **kwargs)
    out = reg.write(reg.entry_for_run(cfg, run_dir))
    print(f"[registry] {out or 'unchanged'}")


# ----------------------------------------------------------------------------- eval


_N_PROMPTS: dict = {}


def n_prompts(blk: dict) -> int:
    from cotcontrol.eval.data import assign_tasks, load_dataset

    key = (blk["dataset"], blk["mode"], blk["split"], blk["max_samples"], blk["subsample_seed"],
           tuple(blk["allowed_modes"] or ()), int(blk["seed"]))
    if key not in _N_PROMPTS:
        samples = load_dataset(blk["dataset"], blk["mode"], None, blk["max_samples"], blk["subsample_seed"], blk["split"])
        _N_PROMPTS[key] = len(assign_tasks(samples, blk["mode"], int(blk["seed"]), allowed_modes=blk["allowed_modes"]))
    return _N_PROMPTS[key]


def block_weight(blk: dict) -> int:
    chunk = int(blk.get("chunk") or 0)
    return (max(1, math.ceil(n_prompts(blk) / chunk)) if chunk else 1) * int(blk["tensor_parallel_size"])


def resolve_checkpoints(cfg: dict, run_dir: Path, steps=None) -> list[tuple[object, str | None]] | None:
    """[(step, lora_path)] to evaluate, or None while train_result.json is missing.
    Sources, in order: cfg['checkpoints'] ([{step, path | 'registry:<run>'}]),
    train_result.json. `steps`: None (all / each block's eval_at_steps), 'final'
    (train_result.final_step), or a list of ints."""
    if cfg.get("checkpoints") is not None:
        ckpts = []
        for c in cfg["checkpoints"]:
            path = c["path"]
            if path.startswith("registry:"):
                path = reg.adapter_path(path.split(":", 1)[1])
            ckpts.append((int(c["step"]), path))
        tr = None
    else:
        tr_path = run_dir / "train_result.json"
        if not tr_path.exists():
            return None
        tr = json.loads(tr_path.read_text())
        ckpts = [(int(c["step"]), c["path"]) for c in tr["checkpoints"]]
    steps = cfg.get("eval_steps") if steps is None else steps
    if steps == "final" and tr is not None:
        final = int(tr.get("final_step", max(s for s, _ in ckpts)))
        ckpts = [(s, p) for s, p in ckpts if s == final]
    elif isinstance(steps, list):
        wanted = {int(s) for s in steps}
        ckpts = [(s, p) for s, p in ckpts if s in wanted]
    return ckpts


def _eval_progress(run_dir: Path) -> dict[tuple, dict]:
    return {(d["block"], str(d["step"])): d for d in read_jsonl(run_dir / "eval_progress.jsonl")}


def write_eval_summary(run_dir: Path, blocks: list[tuple[str, dict]], base_model: str, run_name: str) -> None:
    """train_eval.py's eval_summary.json schema (+ honest_rate) from eval_progress.jsonl."""
    progress = _eval_progress(run_dir)
    evals = {}
    for name, blk in blocks:
        lines = [v for (b, _), v in progress.items() if b == name]
        lines.sort(key=lambda c: (c["step"] == "base", c["step"]))
        evals[name] = {"config": blk, "checkpoints": [{k: v for k, v in ln.items() if k != "block"} for ln in lines]}
    summary = {"run_name": run_name, "base_model": base_model, "evals": evals,
               "checkpoints": evals[blocks[0][0]]["checkpoints"] if blocks else []}
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    try:
        te().plot_summary(summary, run_dir)
    except Exception as e:  # noqa: BLE001 - plotting is cosmetic
        print(f"[summary] plot failed for {run_dir.name}: {e}")
    print(f"[summary] wrote {rel(run_dir / 'eval_summary.json')}")


def eval_jobs(cfg: dict, steps=None, only: list[str] | None = None) -> list[Job]:
    """One job per (eval block, checkpoint); jobs whose checkpoints are unknown yet
    (training in progress) are returned with ready() False."""
    from cotcontrol.eval.eval import eval_cotcontrolqa
    from cotcontrol.inference.modal_vllm import make_generate_fn

    t = te()
    run_dir = run_dir_of(cfg)
    view = train_eval_config(cfg)
    blocks = t.eval_blocks(view.get("eval") or {})
    if only:
        blocks = [(n, b) for n, b in blocks if any(fnmatch.fnmatch(n, p) for p in only)]
    base_model = view["base_model"]

    def tc_of():
        tc, _, _ = _tc_from_cfg(cfg)
        return tc

    def targets(blk):
        ck = resolve_checkpoints(cfg, run_dir, steps)
        if ck is None:
            return None
        if steps is None and blk.get("eval_at_steps") is not None and cfg.get("eval_steps") is None:
            wanted = {int(s) for s in blk["eval_at_steps"]}
            ck = [(s, p) for s, p in ck if s in wanted]
        return ([("base", None)] if blk["eval_base_model"] else []) + ck

    def make(name, blk, step, lora):
        stem = f"checkpoint-{step}" if lora else "base"
        out = run_dir / "eval" / name / f"{stem}.json"

        async def run():
            tc = tc_of()
            gen_cfg = t.build_generate_config(tc, blk)
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            chunk = int(blk.get("chunk") or 0)
            inner = make_generate_fn(base_model, gen_cfg, lora)

            async def gen(messages_list):
                if not chunk:
                    return await inner(messages_list)
                parts = [messages_list[i:i + chunk] for i in range(0, len(messages_list), chunk)]
                outs = await asyncio.gather(*[inner(p) for p in parts])
                return [r for o in outs for r in o]

            result = await eval_cotcontrolqa(
                model=base_model, system_prompt=t.resolve_system_prompt(blk["system_prompt"]), generate_fn=gen,
                save_dir=out.parent, save_name=out.stem, dataset=blk["dataset"], mode=blk["mode"],
                allowed_modes=blk["allowed_modes"], seed=int(blk["seed"]),
                state_requirement=bool(blk["state_requirement"]), max_samples=blk["max_samples"],
                subsample_seed=blk["subsample_seed"], split=blk["split"],
                grade_meta_discussion=blk["grade_meta_discussion"],
                meta_discussion_scope=blk.get("meta_discussion_scope", "compliant"),
                judge_model=blk["judge_model"], judge_concurrency=int(blk["judge_concurrency"]),
                backend_info={"base_model": base_model, "lora_path": lora, "generate_config": dataclasses.asdict(gen_cfg)},
            )
            from kaushik.core.results import honest_rate

            s = result["summary"]
            append_jsonl(run_dir / "eval_progress.jsonl", {
                "block": name, "step": step, "lora_path": lora, "accuracy": s["accuracy"],
                "compliance_rate": s["compliance_rate"], "honest_rate": honest_rate(result),
                "meta_discussion_rate": s["meta_discussion_rate"], "n_errors": s["n_errors"],
                "n_rollouts": s["n_rollouts"], "per_mode": s["per_mode"], "seconds": round(time.time() - t0),
                "time": datetime.now().isoformat()})
            print(f"[done] {run_dir.name} {name} {stem}: compliance={s['compliance_rate']} "
                  f"accuracy={s['accuracy']} errors={s['n_errors']}", flush=True)

        return Job(name=f"eval:{run_dir.name}/{name}/{stem}", weight=block_weight(blk), run=run,
                   done=out.exists, priority=(1, cfg.get("model") or "", 0 if name.endswith("lowercase_thinking") else 1, name),
                   group=str(run_dir))

    def expand(name, blk):
        return [make(name, blk, step, lora) for step, lora in targets(blk)]

    jobs = []
    for name, blk in blocks:
        tg = targets(blk)
        if tg is None:   # training in progress: a placeholder that expands once train_result.json exists
            jobs.append(Job(name=f"eval:{run_dir.name}/{name}/*", weight=int(blk["tensor_parallel_size"]),
                            ready=lambda: (run_dir / "train_result.json").exists(),
                            expand=lambda n=name, b=blk: expand(n, b), group=str(run_dir)))
            continue
        jobs += expand(name, blk)
    return jobs


def eval_main(cfg: dict, argv: list[str]) -> None:
    """Eval blocks of one run (or the base model) through the shared scheduler."""
    a = _common_args("eval", argv,
                     ("--steps", {"nargs": "*", "help": "'final', 'all' or checkpoint steps (default: config eval_steps)"}),
                     ("--blocks", {"nargs": "*", "help": "glob patterns of block names"}),
                     ("--once", {"action": "store_true"}))
    steps = None
    if a.steps:
        steps = a.steps[0] if a.steps[0] in ("final", "all") else [int(s) for s in a.steps]
        steps = None if steps == "all" else steps
    run_dir = run_dir_of(cfg)
    if not a.dry_run:   # a dry run touches nothing in the run dir
        prepare_run(cfg, force=a.force)
    jobs = eval_jobs(cfg, steps, a.blocks)
    view = train_eval_config(cfg)
    blocks = te().eval_blocks(view.get("eval") or {})

    def on_done(group, js):
        write_eval_summary(Path(group), blocks, view["base_model"], view["run_name"])

    run_scheduler(jobs, run_dir.parent, cap=a.cap, name="eval_scheduler", dry_run=a.dry_run, on_group_done=on_done)


# ----------------------------------------------------------------------------- registry


def registry_jobs(cfg: dict) -> list[Job]:
    run_dir = run_dir_of(cfg)
    name = cfg["run_name"]

    async def run():
        out = reg.write(reg.entry_for_run(cfg, run_dir))
        print(f"[registry] {name}: {out or 'unchanged'}")

    return [Job(name=f"registry:{name}", weight=0, run=run, done=lambda: reg.path_for(name).exists(),
                ready=lambda: (run_dir / "train_result.json").exists(), priority=(2, name), group=str(run_dir))]


def registry_main(cfg: dict, argv: list[str]) -> None:
    a = _common_args("registry", argv, ("--step", {"type": int, "default": None}),
                     ("--name", {"default": None, "help": "write under another run_name (e.g. the it-10 exception)"}))
    entry = reg.entry_for_run(cfg, run_dir_of(cfg), final_step=a.step, run_name=a.name)
    if a.dry_run:
        print(json.dumps(entry, indent=2))
        return
    out = reg.write(entry, force=a.force)
    print(f"[registry] {out or 'unchanged (same adapter); --force to rewrite'}")


# ----------------------------------------------------------------------------- status


def status_main(cfg: dict, argv: list[str]) -> None:
    """Training / eval progress of every run dir next to this config's run_dir."""
    ap = argparse.ArgumentParser(prog="status")
    ap.add_argument("--runs-dir", default=None)
    a = ap.parse_args(argv)
    runs_dir = Path(a.runs_dir) if a.runs_dir else run_dir_of(cfg).parent
    print(f"{'run':<44} {'train':<14} {'eval':>7}  last")
    for rd in sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "config.yaml").exists()):
        if (rd / "train_result.json").exists():
            tr = json.loads((rd / "train_result.json").read_text())
            fl = tr.get("final_loss")
            state = f"done {fl:.4f}" if isinstance(fl, (int, float)) else f"done step {tr.get('final_step', '')}"
        elif (rd / "launcher.pid").exists() or (rd / "train_stdout.pid").exists():
            pf = rd / ("launcher.pid" if (rd / "launcher.pid").exists() else "train_stdout.pid")
            state = "running" if pid_alive(int(pf.read_text().strip() or 0)) else "FAILED"
        else:
            state = "-"
        prog = read_jsonl(rd / "eval_progress.jsonl")
        n_done = len({(d["block"], str(d["step"])) for d in prog})
        last = ""
        if prog:
            d = prog[-1]
            last = f"{d['block']}@{d['step']} honest={d.get('honest_rate')} acc={d.get('accuracy')}"
        print(f"{rd.name:<44} {state:<14} {n_done:>7}  {last}")
    for st in sorted(runs_dir.glob("*scheduler.json")):
        s = json.loads(st.read_text())
        alive = pid_alive(int(s.get("pid") or 0))
        print(f"\n{st.stem} {'alive' if alive else 'NOT running'} @ {s['time']}: cap {s['cap']} limit {s.get('limit')} "
              f"hold={s.get('hold') or '-'} in use {s['gpus_in_use']}, {len(s['in_flight'])} in flight, "
              f"{len(s['pending'])} pending, {s['done']} done, {len(s['failed'])} failed")


JOBS = {"train": (train_jobs, train_main), "rl": (rl_jobs, rl_main), "eval": (eval_jobs, eval_main),
        "registry": (registry_jobs, registry_main), "status": (None, status_main)}
