"""One config tree controls every run (kaushik/ARCHITECTURE.md §1).

Layers, each overriding the previous one (lower wins only where higher is silent):

1. `kaushik/config/defaults.yaml`: model table, eval-block protocol defaults,
   Modal / scheduler / path settings.
2. `kaushik/<exp>/config.yaml` (optionally `extends:` another config file).
3. a sweep point: dotted-key overrides from `kaushik/<exp>/sweeps/*.yaml`.
4. `key=value` command-line overrides (values parsed as YAML).

`resolve()` merges 2-4, then *materializes* the tree:

* `model: <key>` pulls `defaults.models[key]`: `base_model`, `gpu`, the `lora`
  targets, the per-block `eval` patch, `train` facts (e.g. attn_implementation)
  and the `train` patches named by `profiles: [reasoning_off, sft]`. These fill
  gaps only; anything the experiment / sweep / CLI set explicitly wins.
* `eval.grid` (or several under `eval.grids: {key: grid}`) expands to named
  blocks: `{conditions, modes, prompts, name, block}` with `prompts` a callable
  reference `path.py:fn(condition, mode, base_model) -> {system_prompt,
  state_requirement}`. Each block = defaults.eval.block < model.eval <
  `eval.block` < the grid's own `block` patch < the grid's keys (`system_prompt`,
  `state_requirement`, `mode`). Named blocks written out by hand get the same
  base (defaults.eval.block < model.eval < `eval.block` < the block itself).
* `modal: {sub: <name>}` selects a deployed sub app pair (`run.py` calls
  `modal_apps.setup_env(sub=...)` before dispatching the job).
* string values whose `{fields}` are all sweep labels (plus `run_name`, `model`,
  `experiment`) are formatted, e.g. `data_path: data/sft_{variant}.jsonl`.

`train_eval_config()` strips the core-only keys (`experiment`, `model`,
`profiles`, `job`, `eval_steps`, `eval.grid`, `eval.block`); what remains is a
plain `scripts/train_eval.py` / `RLConfig` YAML.
"""

from __future__ import annotations

import copy
import string
from functools import cache
from pathlib import Path

import yaml

from kaushik.core.io import KAUSHIK, ROOT, load_callable

DEFAULTS_PATH = KAUSHIK / "config" / "defaults.yaml"
# Core-only top-level keys, stripped before a config reaches train_eval / RLConfig.
# Other sections (meta, task, inference, ...) pass through: Sebastian's loaders ignore
# unknown sections, and experiment jobs read them.
CORE_KEYS = ("experiment", "model", "profiles", "job", "eval_steps", "extends", "hooks", "registry",
             "checkpoints", "analysis", "data", "modal", "eval_sections")
EVAL_META_KEYS = ("grid", "grids", "block")


# ----------------------------------------------------------------------------- tree helpers


def deep_merge(base: dict, over: dict) -> dict:
    """Recursive merge; `over` wins; lists and scalars are replaced, not joined."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def deep_set(d: dict, dotted: str, value) -> None:
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def deep_get(d: dict, dotted: str, default=None):
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def apply_overrides(cfg: dict, point: dict | None) -> dict:
    """Dotted-key overrides ({'train.lr': 1e-4}); `_label` entries are skipped."""
    out = copy.deepcopy(cfg)
    for k, v in (point or {}).items():
        if k == "_label":
            continue
        deep_set(out, k, copy.deepcopy(v))
    return out


def parse_override(arg: str) -> tuple[str, object]:
    """'train.lr=1e-4' -> ('train.lr', 0.0001); values are YAML (so 'null', '[1,2]' work)."""
    if "=" not in arg:
        raise ValueError(f"override must look like key=value: {arg!r}")
    k, v = arg.split("=", 1)
    return k.strip(), yaml.safe_load(v) if v != "" else ""


def parse_overrides(args: list[str]) -> dict:
    return dict(parse_override(a) for a in args)


# ----------------------------------------------------------------------------- loading


@cache
def defaults(path: Path = DEFAULTS_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def model_spec(key: str) -> dict:
    models = defaults()["models"]
    if key not in models:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(models)}")
    return models[key]


def model_key_of(base_model: str) -> str | None:
    """'Qwen/Qwen3-8B' -> 'qwen8b' (None if the id is not in the table)."""
    for k, spec in defaults()["models"].items():
        if spec["base_model"] == base_model:
            return k
    return None


def experiment_dir(exp: str) -> Path:
    return KAUSHIK / exp


def load_config_file(path: Path) -> dict:
    """YAML with optional `extends: <relative path>` (parent merged under child)."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent:
        return deep_merge(load_config_file((path.parent / parent).resolve()), raw)
    return raw


def load_experiment(exp: str) -> dict:
    path = experiment_dir(exp) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config.yaml for experiment {exp!r} ({path})")
    return deep_merge({"experiment": exp}, load_config_file(path))


# ----------------------------------------------------------------------------- materialization


def _format_strings(obj, fields: dict):
    """Format str values whose every {field} is known; leave the rest untouched."""
    if isinstance(obj, dict):
        return {k: _format_strings(v, fields) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_format_strings(v, fields) for v in obj]
    if isinstance(obj, str) and "{" in obj:
        try:
            names = [f for _, f, _, _ in string.Formatter().parse(obj) if f is not None]
        except ValueError:
            return obj
        if names and all(n in fields for n in names):
            return obj.format(**fields)
    return obj


def _model_base(spec: dict, profiles: list[str]) -> dict:
    train = dict(spec.get("train") or {})
    for p in profiles:
        patch = (spec.get("profiles") or {}).get(p)
        if patch is None:
            raise KeyError(f"model {spec['base_model']} has no profile {p!r}; "
                           f"known: {sorted(spec.get('profiles') or {})}")
        train = deep_merge(train, patch)
    base = {"base_model": spec["base_model"], "train": train}
    if "gpu" in spec:
        base["gpu"] = spec["gpu"]
    if spec.get("lora"):
        base["lora"] = spec["lora"]
    return base


def expand_grid(grid: dict, block: dict, base_model: str) -> dict:
    """{conditions, modes, prompts, name} -> {block_name: fully specified block}."""
    prompts = load_callable(grid["prompts"])
    name_fmt = grid.get("name", "{condition}_{mode}")
    blocks = {}
    for mode in grid["modes"]:
        for cond in grid["conditions"]:
            p = prompts(cond, mode, base_model)
            blk = deep_merge(block, {"system_prompt": p.get("system_prompt") or "",
                                     "state_requirement": bool(p.get("state_requirement", True)),
                                     "mode": mode})
            blocks[name_fmt.format(condition=cond, mode=mode)] = blk
    return blocks


def eval_sections(cfg: dict) -> list[str]:
    return list(cfg.get("eval_sections") or ["eval"])


def expand_eval_section(ev: dict, spec: dict | None, base_model: str) -> dict:
    """One eval section -> {block_name: fully specified block}: base = defaults.eval.block
    < model.eval < `block`; named blocks = base < own keys (a `null` entry drops an
    inherited block); grids expand with their own `block` patch on top of base."""
    base = deep_merge(defaults()["eval"]["block"], spec.get("eval", {}) if spec else {})
    base = deep_merge(base, ev.get("block") or {})
    named = {k: deep_merge(base, v) for k, v in ev.items() if k not in EVAL_META_KEYS and isinstance(v, dict)}
    grids = ({"_": ev["grid"]} if "grid" in ev else {}) | dict(ev.get("grids") or {})
    for g in grids.values():   # per-grid `block:` patch applies to that grid only
        named = {**expand_grid(g, deep_merge(base, g.get("block") or {}), base_model), **named}
    dropped = {k for k, v in ev.items() if k not in EVAL_META_KEYS and v is None}
    return {k: v for k, v in named.items() if k not in dropped}


def materialize(cfg: dict, labels: dict | None = None) -> dict:
    """Apply the model entry, expand the eval grid and format strings (see module doc)."""
    cfg = copy.deepcopy(cfg)
    spec = model_spec(cfg["model"]) if cfg.get("model") else None
    if spec is not None:
        cfg = deep_merge(_model_base(spec, list(cfg.get("profiles") or [])), cfg)
    cfg["analysis"] = deep_merge(defaults()["analysis"], cfg.get("analysis") or {})
    # Blocks are expanded only once the model is known; a config resolved without one
    # (analysis jobs) keeps `eval.grid(s)` / `eval.block` as data for the job to read.
    # `eval_sections` (default [eval]) lists every section expanded this way, e.g. a
    # `final_eval` section of test-split blocks read by the experiment's own eval code.
    for section in eval_sections(cfg):
        ev = cfg.get(section)
        if isinstance(ev, dict) and any(k in ev for k in EVAL_META_KEYS) and cfg.get("base_model"):
            cfg[section] = expand_eval_section(ev, spec, cfg["base_model"])
    fields = {**(labels or {}), "model": cfg.get("model"), "run_name": cfg.get("run_name"),
              "experiment": cfg.get("experiment")}
    return _format_strings(cfg, {k: v for k, v in fields.items() if v is not None})


def resolve(exp: str, point: dict | None = None, overrides: dict | None = None,
            labels: dict | None = None, run_name: str | None = None) -> dict:
    """Experiment config + sweep point + overrides, materialized."""
    cfg = load_experiment(exp)
    cfg = apply_overrides(cfg, point)
    cfg = apply_overrides(cfg, overrides)
    if run_name:
        cfg["run_name"] = run_name
    return materialize(cfg, labels)


def train_eval_config(cfg: dict) -> dict:
    """The resolved tree minus core-only keys: what scripts/train_eval.py / RLConfig read."""
    out = {k: v for k, v in cfg.items() if k not in CORE_KEYS}
    for section in eval_sections(cfg):
        if isinstance(out.get(section), dict):
            out[section] = {k: v for k, v in out[section].items() if k not in EVAL_META_KEYS and v is not None}
    return out


def run_dir_of(cfg: dict) -> Path:
    if "run_dir" not in cfg:
        raise KeyError("config has no run_dir")
    return ROOT / cfg["run_dir"]


def runs_root(cfg: dict) -> Path:
    """The folder holding this experiment's run dirs (parent of the run_dir template,
    e.g. kaushik/orange_expand/sft/runs), usable before any run name is known."""
    return (ROOT / cfg["run_dir"]).parent


def results_dir(cfg: dict) -> Path:
    return experiment_dir(cfg["experiment"]) / defaults()["paths"]["results"]


def dump_yaml(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, width=1000, allow_unicode=False)
