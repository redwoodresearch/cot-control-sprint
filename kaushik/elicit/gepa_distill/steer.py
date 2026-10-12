"""Steering-vector pipeline driver (local side; remote code in steer_app.py). Every knob comes from the
`steer:` section of config.yaml (override with key=value, e.g. `steer.build.layers=[12]`).

    run.py elicit/gepa_distill steer extract [steer.extract.prompt=brevity steer.extract.name=vbrief]
    run.py elicit/gepa_distill steer build  [steer.build.kind=resid steer.build.proj=vbrief steer.build.layers=[12,15]]
    run.py elicit/gepa_distill steer build steer.build.kind=random steer.build.layers=[15] steer.build.alphas=[2]
    run.py elicit/gepa_distill steer cleanup --variants last_L12_a1,...      # delete the 13 GB checkpoints
    run.py sweep elicit/gepa_distill/sweeps/steer_sweep.yaml --dry-run        # the 40-question screen of built variants

extract: train-split prompt pairs (with / without the system prompt) -> mean residual-stream differences per
layer (`last` token and `suffix_mean`), saved as runs/steer/vectors_<name>.json (+ .pt on the volume).
build: one steered copy of the HF checkpoint per (layer, alpha) with o_proj.bias += alpha * v, recorded in
runs/steer/ckpts.json; variant naming <kind>_L<layer>_a<alpha>[_s<seed>].
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from kaushik.core.io import atomic_write_json, read_json
from kaushik.elicit.common import app_name, check_modal, runs_dir, setup_modal
from kaushik.elicit.common.distill import teacher_prompt


def variant_name(kind: str, layer: int, alpha: float, seed: int | None = None) -> str:
    """<kind>_L<layer>_a<alpha>[_s<seed>]; alpha is the multiple of ||v_gepa_last[layer]|| for random/resid too."""
    return f"{kind}_L{layer}_a{alpha:g}" + (f"_s{seed}" if kind == "random" else "")


def ckpt_path(cfg: dict, variant: str) -> str:
    return f"{cfg['steer']['vol_dir']}/steer/ckpt/{variant}"


def steer_prompt(cfg: dict, key: str) -> str:
    """`steer.prompts.<key>`: null = the GEPA prompt pinned in distill.teacher, else the literal."""
    prompts = cfg["steer"]["prompts"]
    if key not in prompts:
        raise KeyError(f"unknown steer prompt {key!r}; known: {sorted(prompts)}")
    return teacher_prompt(cfg) if prompts[key] is None else prompts[key]


def train_pairs(cfg: dict, n: int, seed: int, system_prompt: str) -> list[dict]:
    """n train-split questions with a random training-mode constraint each (never val/test)."""
    from cotcontrol.eval.data import assign_tasks, load_dataset
    from cotcontrol.eval.prompts import create_user_prompt

    samples = load_dataset("all", "random", split="train", max_samples=n, subsample_seed=seed)
    tasks = assign_tasks(samples, "random", seed=seed)
    pairs = []
    for s, m in tasks:
        user = create_user_prompt(s, m, cfg["base_model"])
        pairs.append({"with": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
                      "without": [{"role": "user", "content": user}], "id": f"{s['dataset']}:{s['id']}", "mode": m})
    return pairs


def extract(cfg: dict, steer_runs) -> None:
    import modal

    ex = cfg["steer"]["extract"]
    pairs = train_pairs(cfg, int(ex["n"]), int(ex["seed"]), steer_prompt(cfg, ex["prompt"]))
    print(f"[extract] {len(pairs)} train pairs (system prompt: {ex['prompt']}); calling {app_name(cfg, 'steer')}.extract", flush=True)
    fn = modal.Function.from_name(app_name(cfg, "steer"), "extract")
    stats = fn.remote([{k: p[k] for k in ("with", "without")} for p in pairs], ex["name"],
                      (int(ex["verify_layer"]), float(ex["verify_alpha"])))
    stats["pairs"] = [{"id": p["id"], "mode": p["mode"]} for p in pairs]
    stats["seed"] = int(ex["seed"])
    stats["system_prompt"] = "GEPA" if cfg["steer"]["prompts"][ex["prompt"]] is None else cfg["steer"]["prompts"][ex["prompt"]]
    stats["timestamp"] = datetime.now().astimezone().isoformat()
    out = steer_runs / f"vectors_{ex['name']}.json"
    atomic_write_json(stats, out, indent=1)
    print(json.dumps({k: stats[k] for k in ("n", "suffix_len", "prompt_len_with", "prompt_len_without", "verify")}, indent=1))
    for i in range(stats["n_layers"]):
        print(f"  L{i:2d}  |v_last|={stats['vec_norm']['last'][i]:7.2f}  |v_suffix|={stats['vec_norm']['suffix_mean'][i]:7.2f}  "
              f"|resid_last| without={stats['resid_norm_last']['without'][i]:7.1f} with={stats['resid_norm_last']['with'][i]:7.1f}")
    print(f"-> {out}")


def build(cfg: dict, steer_runs) -> None:
    import modal

    b = cfg["steer"]["build"]
    fn = modal.Function.from_name(app_name(cfg, "steer"), "build_checkpoint")
    specs = [(int(L), float(a)) for L in b["layers"] for a in b["alphas"]]
    calls = {}
    for L, a in specs:
        v = variant_name(b["kind"], L, a, int(b["seed"]))
        calls[v] = fn.spawn(b["vec"], b["kind"], L, a, v, int(b["seed"]), b.get("proj"), cfg["steer"]["hf_snapshot"])
    print(f"[build] spawned {len(calls)} builds: {list(calls)}", flush=True)
    rec_path = steer_runs / "ckpts.json"
    recs = read_json(rec_path, {})
    for v, c in calls.items():
        info = c.get()
        recs[v] = {**info, "variant": v, "base_model": ckpt_path(cfg, v), "built": datetime.now().astimezone().isoformat()}
        print(f"[build] {v}: norm {info['vector_norm']:.2f} ({info['build_seconds']}s)", flush=True)
        atomic_write_json(recs, rec_path, indent=1)


def cleanup(cfg: dict, variants: list[str]) -> None:
    import modal

    fn = modal.Function.from_name(app_name(cfg, "steer"), "remove_path")
    for v in variants:
        print(v, fn.remote(ckpt_path(cfg, v)))


def main(cfg: dict, argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="steer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "build", "cleanup"])
    ap.add_argument("--variants", default=None, help="cleanup: comma-separated variant names")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    steer_runs = runs_dir(cfg) / "steer"
    st = cfg["steer"]
    if a.dry_run:
        if a.cmd == "extract":
            print(f"extract: {st['extract']} -> {steer_runs / ('vectors_' + st['extract']['name'] + '.json')}")
        elif a.cmd == "build":
            b = st["build"]
            print("build:", [variant_name(b["kind"], int(L), float(x), int(b["seed"])) for L in b["layers"] for x in b["alphas"]])
        else:
            print("cleanup:", [ckpt_path(cfg, v) for v in (a.variants or "").split(",") if v])
        return
    setup_modal(cfg)
    check_modal(cfg)
    steer_runs.mkdir(parents=True, exist_ok=True)
    if a.cmd == "extract":
        extract(cfg, steer_runs)
    elif a.cmd == "build":
        build(cfg, steer_runs)
    else:
        if not a.variants:
            raise SystemExit("cleanup needs --variants a,b,...")
        cleanup(cfg, a.variants.split(","))
