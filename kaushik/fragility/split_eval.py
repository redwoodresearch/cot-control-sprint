"""Parallel eval workers for a train_eval.py run whose training is done.

train_eval.py evaluates its blocks one after another, so a Qwen3.6-27B run
(23 blocks x ~15 min) takes ~6 h. This launches K extra workers, each running
`scripts/train_eval.py --eval-only` on a helper dir
(<run_dir>/_workers/w<k>/: config.yaml with a disjoint subset of the not-yet-
done blocks + a copy of train_result.json). cotcontrol.inference.modal_vllm
caches every generation on disk keyed by adapter path + prompt + sampling
params, so when the main process reaches those blocks it gets cache hits and
only re-grades. The main run dir stays the single source of truth; the workers'
own eval JSONs are kept under _workers/ as artifacts. Workers take blocks from
the END of the block order (tail-first), the main process from the front.

    .venv/bin/python kaushik/run.py fragility split_eval --run fragility-mcq_noreason-qwen36_27b-seed0 --workers 2 [--dry-run]

Each worker holds the config's peak GPUs (parallel engine calls) — account for
them against the sprint cap by hand (the scheduler only counts its own launches).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from datetime import datetime

import yaml
from common import PY, ROOT, RUNS, atomic_write_json

N_TEST = {"final_modes": 4500, "final_heldout": 1500}  # prompts per checkpoint; other test blocks 500, val 200


def block_plan(raw: dict, run_dir) -> tuple[list[str], list[str], dict[str, int]]:
    """-> (done, pending-for-workers, estimated generations per block). The first
    unfinished block in config order is assumed to be the main process' current one."""
    n_ckpt = len(raw["train"]["save_at_steps"]) + 1
    done, todo, est = [], [], {}
    for name, blk in raw["eval"].items():
        steps = len(blk["eval_at_steps"]) if blk.get("eval_at_steps") else n_ckpt
        have = len(list((run_dir / "eval" / name).glob("*.json"))) if (run_dir / "eval" / name).is_dir() else 0
        per = N_TEST.get(name, 500) if blk.get("split") == "test" else 200
        est[name] = per * steps
        (done if have >= steps else todo).append(name)
    return done, todo[1:], est


def assign(blocks: list[str], est: dict[str, int], k: int) -> list[list[str]]:
    """Longest-processing-time balance over k workers; each list tail-first."""
    order = {b: i for i, b in enumerate(blocks)}
    loads, groups = [0] * k, [[] for _ in range(k)]
    for b in sorted(blocks, key=lambda b: -est[b]):
        i = loads.index(min(loads))
        loads[i] += est[b]
        groups[i].append(b)
    return [sorted(g, key=lambda b: -order[b]) for g in groups if g]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run name under kaushik/fragility/runs/")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    run_dir = RUNS / args.run
    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    if not (run_dir / "train_result.json").exists():
        raise SystemExit(f"{args.run}: training not finished (no train_result.json)")
    done, todo, est = block_plan(raw, run_dir)
    groups = assign(todo, est, args.workers)
    print(f"{args.run}: {len(done)} blocks done, {len(todo)} assigned to {len(groups)} workers "
          f"(~{sum(est[b] for b in todo)} generations)")
    for k, blocks in enumerate(groups):
        print(f"  w{k}: {sum(est[b] for b in blocks)} gens: {blocks}")
    if args.dry_run or not groups:
        return
    for k, blocks in enumerate(groups):
        helper = run_dir / "_workers" / f"w{k}"
        if (helper / "manifest.json").exists():
            print(f"  w{k}: already launched ({helper}); skipping")
            continue
        helper.mkdir(parents=True, exist_ok=True)
        cfg = {**raw, "run_dir": str(helper.relative_to(ROOT)), "eval": {b: raw["eval"][b] for b in blocks}}
        (helper / "config.yaml").write_text(yaml.dump(cfg, sort_keys=False, width=1000))
        shutil.copy2(run_dir / "train_result.json", helper / "train_result.json")
        cmd = [PY, "scripts/train_eval.py", "--eval-only", str(helper.relative_to(ROOT))]
        with open(helper / "launcher.log", "a") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=os.environ.copy())
        atomic_write_json({"run": args.run, "worker": k, "blocks": blocks, "pid": proc.pid,
                           "command": " ".join(cmd), "launched_at": datetime.now().astimezone().isoformat()},
                          helper / "manifest.json")
        print(f"  launched w{k} (pid {proc.pid}) -> {helper.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
