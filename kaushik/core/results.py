"""Result loaders: per-rollout rows from train_eval-layout eval JSONs, keyed by
(dataset, id, mode) — question ids repeat across datasets, so `qid` is
"<dataset>:<id>" and `task` is "<qid>|<mode>".

    rows = parse_eval_json(path)                    # one dict per non-error sample
    df = load_rollouts({"sft": runs_dir, ...}, cache_dir, extra=my_row_fn)

Each row: source, run, block, step ("base" or the checkpoint step as a string),
lora, base_model, system_prompt, state_requirement, qid, task, mode, compliance,
honest (cotcontrol.eval.grading.is_honest), correct, chars and words (graded
reasoning text), truncated. `extra(row, data_config, path)` lets an experiment add arm /
seed / condition columns; the cache stores the final rows and is invalidated by
the JSON's mtime (`cache_version` bumps it).

Helpers: `infer_condition` (match a block's system prompt against the experiment's
prompt builder), `run_meta`, `parse_run_name`, `final_step`, `honest_rate`.
"""

from __future__ import annotations

import json
import pickle
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import yaml

from cotcontrol.eval.grading import is_honest

RUN_NAME_RE = re.compile(r"^(?P<arm>.+?)-(?P<model>[A-Za-z0-9_.]+)-seed(?P<seed>\d+)(?P<rest>.*)$")
TIMESTAMP_RE = re.compile(r"-\d{8}_\d{6}$")


def honest_rate(result: dict) -> float | None:
    samples = [s for rec in result["results"] for s in rec["samples"] if not s["error"]]
    return sum(is_honest(s) for s in samples) / len(samples) if samples else None


def parse_eval_json(path: Path, source: str = "", extra: Callable[[dict, dict, Path], dict] | None = None) -> list[dict]:
    path = Path(path)
    data = json.loads(path.read_text())
    cfg = data["config"]
    run, block = path.parents[2].name, path.parent.name
    base = {
        "source": source, "run": run, "block": block, "step": path.stem.replace("checkpoint-", ""),
        "lora": (cfg.get("backend_info") or {}).get("lora_path"), "base_model": cfg["model"],
        "system_prompt": cfg.get("system_prompt") or "", "state_requirement": cfg.get("state_requirement", True),
    }
    rows = []
    for rec in data["results"]:
        mode = rec["mode"]
        qid = f"{rec['dataset']}:{rec['id']}"
        for s in rec["samples"]:
            if s["error"]:
                continue
            text = s.get("reasoning_text_graded") or ""
            row = {**base, "qid": qid, "task": f"{qid}|{mode}", "mode": mode,
                   "compliance": s["compliance"] == 1, "honest": is_honest(s), "correct": s["correct"] is True,
                   "chars": len(text), "words": len(text.split()), "truncated": s.get("finish_reason") == "length"}
            if extra is not None:
                row.update(extra(row, cfg, path))
            rows.append(row)
    return rows


def load_rollouts(sources: dict[str, Path], cache_dir: Path, extra=None, cache_version: str = "v3",
                  patterns: tuple[str, ...] = ("*/eval/*/*.json", "*/*/eval/*/*.json"),
                  skip: str = "smoke") -> pd.DataFrame:
    """Every eval JSON under each source root (two layouts: runs/<run>/eval/<block>/
    and runs/<group>/<run>/eval/<block>/), cached per file by mtime."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, root in sources.items():
        root = Path(root)
        if not root.is_dir():
            print(f"[load] no {source} runs at {root}")
            continue
        paths = sorted({p for pat in patterns for p in root.glob(pat)})
        for path in paths:
            relp = path.relative_to(root)
            if skip and skip in str(relp):
                continue
            cache = cache_dir / f"{cache_version}__{source}__{'__'.join(relp.parts[:-3])}__{path.parent.name}__{path.stem}.pkl"
            mtime = path.stat().st_mtime
            if cache.exists():
                c = pickle.loads(cache.read_bytes())
                if c["mtime"] == mtime:
                    rows += c["rows"]
                    continue
            try:
                parsed = parse_eval_json(path, source, extra)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"[load] skipping {path}: {type(e).__name__}: {e}")
                continue
            cache.write_bytes(pickle.dumps({"mtime": mtime, "rows": parsed}))
            rows += parsed
    return pd.DataFrame(rows)


def infer_condition(system_prompt: str, state_requirement: bool, mode: str, base_model: str,
                    prompts: Callable[[str, str, str], dict], conditions: list[str],
                    block: str | None = None, stated: str = "standard") -> str:
    """Name the eval condition from the JSON's own prompt: empty prompt -> `stated`
    (or "bare" without a stated requirement); otherwise the condition whose
    `prompts(condition, mode, base_model)["system_prompt"]` matches; then the block
    name's prefix; else "other"."""
    sp = (system_prompt or "").strip()
    if not sp:
        return stated if state_requirement else "bare"
    for cond in conditions:
        if cond == stated:
            continue
        if sp == (prompts(cond, mode, base_model).get("system_prompt") or "").strip():
            return cond
    if block:
        m = re.match(r"^(" + "|".join(map(re.escape, conditions)) + r")_", block)
        if m:
            return m.group(1)
    return "other"


def run_meta(run_dir: Path) -> dict:
    cfg = Path(run_dir) / "config.yaml"
    if cfg.exists():
        try:
            return (yaml.safe_load(cfg.read_text()) or {}).get("meta") or {}
        except yaml.YAMLError:
            return {}
    return {}


def parse_run_name(name: str) -> dict | None:
    """'oex-sft_eugene-qwen8b-seed0-lr5e-5' -> {arm: 'oex-sft_eugene', model: 'qwen8b',
    seed: 0, rest: '-lr5e-5'}; Modal timestamps are stripped first."""
    m = RUN_NAME_RE.match(TIMESTAMP_RE.sub("", name))
    return {"arm": m["arm"], "model": m["model"], "seed": int(m["seed"]), "rest": m["rest"]} if m else None


def final_step(run_dir: Path) -> int:
    tr = json.loads((Path(run_dir) / "train_result.json").read_text())
    return int(tr.get("final_step", tr.get("iterations_completed", max(int(c["step"]) for c in tr["checkpoints"]))))
