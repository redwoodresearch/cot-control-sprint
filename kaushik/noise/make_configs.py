"""Group registry entries into noise jobs: one config per (experiment, arm, base_model).

    .venv/bin/python kaushik/run.py noise make_configs

Reads kaushik/registry/*.json (final adapters announced by the training agents)
and writes configs/<group>.yaml. A group pools the arm's seeds: its 3 noise
adapters are calibrated to the MEDIAN KL(base || adapter) of the trained seeds
and evaluated once with the arm's final eval blocks, instead of 3 noise evals
per trained seed. `watch` re-runs this every poll (idempotent; entries only
accumulate).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
KAUSHIK = HERE.parent
ROOT = KAUSHIK.parent
REGISTRY = KAUSHIK / "registry"
CONFIGS = HERE / "configs"

NOISE_SEEDS = [0, 1, 2]
KL_SETTINGS = {"split": "val", "dataset": "all", "n_prompts": 200, "max_tokens": 2048, "seed": 0}
EXPERIMENTS = ("fragility", "fragility/cpt", "orange", "orange/rl")
KL_ONLY_GROUP_KEYS = ("lowlr", "cpt_param_sweep", "cptk")


def model_slug(base_model: str) -> str:
    """'openai/gpt-oss-20b' -> 'gptoss20b', 'Qwen/Qwen3.6-27B' -> 'qwen3627b'."""
    return re.sub(r"[^a-z0-9]", "", base_model.split("/")[-1].lower())


def load_entries(registry_dir: Path = REGISTRY) -> list[dict]:
    """Registry entries with the required keys (partial / malformed files skipped)."""
    required = {"run_name", "experiment", "arm", "base_model", "seed", "adapter_path", "config_path"}
    out = []
    for p in sorted(registry_dir.glob("*.json")):
        try:
            e = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(e, dict) or not required <= set(e):
            continue
        if "smoke" in str(e["run_name"]):  # other folders' smoke runs are not arms
            continue
        e["_registry_path"] = str(p.relative_to(ROOT))
        out.append(e)
    return out


def run_variant(run_name: str) -> str:
    """Suffix after '-seedN' (e.g. 'step50' for cpt's matched-dose checkpoint): a
    different dose of the same arm must not share a group / median KL."""
    m = re.search(r"[-_]seed\d+[-_]?(.*)$", run_name)
    return (m.group(1) if m else "").strip("-_")


def group_name(e: dict) -> str:
    arm = e["arm"] + (f"-{run_variant(e['run_name'])}" if run_variant(e["run_name"]) else "")
    return f"{e['experiment'].replace('/', '-')}-{arm}-{model_slug(e['base_model'])}"


def run_prefix(run_name: str) -> str:
    return re.sub(r"[-_]seed\d+.*$", "", run_name)


def expected_seeds(entries: list[dict]) -> int:
    """How many seeds the arm will register: an explicit n_seeds, else the number
    of matching configs in the owning experiment folder, else the spec default
    (5 on Qwen3.6-27B, 3 otherwise)."""
    for e in entries:
        if e.get("n_seeds"):
            return int(e["n_seeds"])
    e = entries[0]
    prefix = run_prefix(e["run_name"])
    cfg_dir = KAUSHIK / e["experiment"] / "configs"
    n = len([p for p in cfg_dir.glob("*.yaml") if run_prefix(p.stem) == prefix and "smoke" not in p.stem]) if cfg_dir.is_dir() else 0
    # Never below the spec default: a sweep cell whose extra seeds are not yet
    # written would otherwise be processed after its first seed (the 4h wait in
    # `watch` still covers arms that genuinely have fewer seeds).
    return max(n, 5 if "3.6" in e["base_model"] else 3, len(entries))


def group_entries(entries: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(group_name(e), []).append(e)
    return {g: sorted(v, key=lambda e: int(e["seed"])) for g, v in groups.items()}


def group_config(name: str, members: list[dict]) -> dict:
    e0 = members[0]
    return {
        "group": name,
        "experiment": e0["experiment"],
        "arm": e0["arm"],
        "base_model": e0["base_model"],
        # Single-seed sweep points (coordinator 14:58): KL only, never a noise group.
        "noise": any(bool(e.get("noise")) for e in members) and not any(k in name for k in KL_ONLY_GROUP_KEYS),
        "measure_kl": any(bool(e.get("measure_kl") or e.get("noise")) for e in members),
        "eval_chat_template_kwargs": e0.get("eval_chat_template_kwargs"),
        "expected_seeds": expected_seeds(members),
        "entries": [{k: e[k] for k in ("run_name", "seed", "adapter_path", "config_path")}
                    | {k: e[k] for k in ("noise_eval_blocks", "eval_chat_template_kwargs") if k in e}
                    | {"registry_path": e["_registry_path"]} for e in members],
        "noise_seeds": list(NOISE_SEEDS),
        "family": e0.get("noise_family", "frob"),
        "tolerance": 0.1,
        "kl": dict(KL_SETTINGS),
        "out_root": f"/checkpoints/noise/{name}",
    }


def write_group_configs(entries: list[dict] | None = None, out_dir: Path = CONFIGS) -> dict[str, dict]:
    entries = load_entries() if entries is None else entries
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = {}
    for name, members in group_entries(entries).items():
        cfg = group_config(name, members)
        path = out_dir / f"{name}.yaml"
        text = yaml.dump(cfg, sort_keys=False, width=1000)
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
        configs[name] = cfg
    return configs


def expected_run_names() -> dict[str, str]:
    """run_name -> experiment for every config the training folders have written."""
    out = {}
    for exp in EXPERIMENTS:
        for p in (KAUSHIK / exp / "configs").glob("*.yaml"):
            try:
                raw = yaml.safe_load(p.read_text()) or {}
            except yaml.YAMLError:
                continue
            if isinstance(raw, dict) and raw.get("run_name") and "smoke" not in str(raw["run_name"]):
                out[str(raw["run_name"])] = exp
    return out


if __name__ == "__main__":
    cfgs = write_group_configs()
    for name, cfg in cfgs.items():
        print(f"{name}: {len(cfg['entries'])}/{cfg['expected_seeds']} seeds, noise={cfg['noise']}")
    print(f"{len(cfgs)} groups -> {CONFIGS}")
