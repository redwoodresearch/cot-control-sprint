"""Steering-vector pipeline driver (local side; remote code in steer_app.py).

    kaushik/run.py elicit/gepa_distill steer extract --n 300 --name v1          # train-split prompt pairs -> vectors
    kaushik/run.py elicit/gepa_distill steer build --vec v1 --kind last --layers 8,12,16 --alphas 1,2,4
    kaushik/run.py elicit/gepa_distill steer build --vec v1 --kind random --layers 12 --alphas 2 --seed 0
    kaushik/run.py elicit/gepa_distill steer sweep --variants last_L8_a1,... [--max-questions 40]   # small val subset evals
    kaushik/run.py elicit/gepa_distill steer cleanup --variants ...                                 # delete 13 GB checkpoints

Variant naming: <kind>_L<layer>_a<alpha>[_s<seed>]; checkpoints at /checkpoints/gepa_distill/steer/ckpt/<variant>,
records in runs/steer/{vectors_<name>.json, ckpts.json}.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime

import modal

from gd_common import (BASE_MODEL, INFERENCE_APP, PY, ROOT, RUNS, STEER_APP, TRAIN_APP, VOL_DIR, atomic_write_json,
                       gepa_prompt, load_json)

sys.path.insert(0, str(ROOT))
from cotcontrol.eval.data import assign_tasks, load_dataset  # noqa: E402
from cotcontrol.eval.prompts import create_user_prompt  # noqa: E402

STEER_RUNS = RUNS / "steer"


def variant_name(kind: str, layer: int, alpha: float, seed: int | None = None) -> str:
    """<kind>_L<layer>_a<alpha>[_s<seed>]; alpha is the multiple of ||v_gepa_last[layer]|| for random/resid too."""
    a = f"{alpha:g}"
    return f"{kind}_L{layer}_a{a}" + (f"_s{seed}" if kind == "random" else "")


def ckpt_path(variant: str) -> str:
    return f"{VOL_DIR}/steer/ckpt/{variant}"


def train_pairs(n: int, seed: int, system_prompt: str | None = None) -> list[dict]:
    """n train-split questions with a random training-mode constraint each (never val/test).
    system_prompt None = the GEPA prompt; else a literal (e.g. the brevity prompt)."""
    samples = load_dataset("all", "random", split="train", max_samples=n, subsample_seed=seed)
    tasks = assign_tasks(samples, "random", seed=seed)
    sp = gepa_prompt() if system_prompt is None else system_prompt
    pairs = []
    for s, m in tasks:
        user = create_user_prompt(s, m, BASE_MODEL)
        pairs.append({"with": [{"role": "system", "content": sp}, {"role": "user", "content": user}],
                      "without": [{"role": "user", "content": user}], "id": f"{s['dataset']}:{s['id']}", "mode": m})
    return pairs


def extract(args) -> None:
    pairs = train_pairs(args.n, args.seed, args.system_prompt)
    print(f"[extract] {len(pairs)} train pairs (system prompt: {'GEPA' if args.system_prompt is None else args.system_prompt!r}); "
          f"calling {STEER_APP}.extract", flush=True)
    fn = modal.Function.from_name(STEER_APP, "extract")
    stats = fn.remote([{k: p[k] for k in ("with", "without")} for p in pairs], args.name,
                      (args.verify_layer, args.verify_alpha))
    stats["pairs"] = [{"id": p["id"], "mode": p["mode"]} for p in pairs]
    stats["seed"] = args.seed
    stats["system_prompt"] = "GEPA" if args.system_prompt is None else args.system_prompt
    stats["timestamp"] = datetime.now().astimezone().isoformat()
    out = STEER_RUNS / f"vectors_{args.name}.json"
    atomic_write_json(stats, out)
    print(json.dumps({k: stats[k] for k in ("n", "suffix_len", "prompt_len_with", "prompt_len_without", "verify")}, indent=1))
    for i in range(stats["n_layers"]):
        print(f"  L{i:2d}  |v_last|={stats['vec_norm']['last'][i]:7.2f}  |v_suffix|={stats['vec_norm']['suffix_mean'][i]:7.2f}  "
              f"|resid_last| without={stats['resid_norm_last']['without'][i]:7.1f} with={stats['resid_norm_last']['with'][i]:7.1f}")
    print(f"-> {out}")


def build(args) -> None:
    fn = modal.Function.from_name(STEER_APP, "build_checkpoint")
    specs = [(int(L), float(a)) for L in args.layers.split(",") for a in args.alphas.split(",")]
    calls = {}
    for L, a in specs:
        v = variant_name(args.kind, L, a, args.seed)
        calls[v] = fn.spawn(args.vec, args.kind, L, a, v, args.seed, args.proj)
    print(f"[build] spawned {len(calls)} builds: {list(calls)}", flush=True)
    rec_path = STEER_RUNS / "ckpts.json"
    recs = load_json(rec_path) if rec_path.exists() else {}
    for v, c in calls.items():
        info = c.get()
        recs[v] = {**info, "variant": v, "base_model": ckpt_path(v), "built": datetime.now().astimezone().isoformat()}
        print(f"[build] {v}: norm {info['vector_norm']:.2f} ({info['build_seconds']}s)", flush=True)
        atomic_write_json(recs, rec_path)


def sweep(args) -> None:
    """Launch one eval process per variant on the small val subset (modes block only unless --blocks)."""
    recs = load_json(STEER_RUNS / "ckpts.json")
    groups = []
    for v in args.variants.split(","):
        if v not in recs:
            raise SystemExit(f"{v} not built (known: {list(recs)})")
        run = f"{args.prefix}{v}"
        g = [run, "--split", args.split, "--base-model", recs[v]["base_model"], "--blocks", args.blocks, "--chunk", str(args.chunk)]
        if args.max_questions:
            g += ["--max-questions", str(args.max_questions), "--subsample-seed", str(args.subsample_seed)]
        groups.append(g)
    argv = [x for i, g in enumerate(groups) for x in (([";;"] if i else []) + g)]
    subprocess.run([PY, str(ROOT / "kaushik" / "run.py"), "elicit/gepa_distill", "launch", *argv], cwd=ROOT, check=True,
                   env={**__import__("os").environ, "COTCONTROL_MODAL_INFERENCE_APP": INFERENCE_APP,
                        "COTCONTROL_MODAL_TRAIN_APP": TRAIN_APP})


def cleanup(args) -> None:
    fn = modal.Function.from_name(STEER_APP, "remove_path")
    for v in args.variants.split(","):
        print(v, fn.remote(ckpt_path(v)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gepa_distill steer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract"); p.add_argument("--n", type=int, default=300); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--name", default="v1"); p.add_argument("--verify-layer", type=int, default=12)
    p.add_argument("--verify-alpha", type=float, default=2.0)
    p.add_argument("--system-prompt", default=None, help="literal system prompt instead of GEPA (brevity direction)")
    p.set_defaults(fn=extract)
    p = sub.add_parser("build"); p.add_argument("--vec", default="v1"); p.add_argument("--kind", default="last")
    p.add_argument("--layers", required=True); p.add_argument("--alphas", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--proj", default=None, help="kind=resid: vector name whose 'last' direction is projected out (brevity)")
    p.set_defaults(fn=build)
    p = sub.add_parser("sweep"); p.add_argument("--variants", required=True); p.add_argument("--split", default="val")
    p.add_argument("--blocks", default="modes"); p.add_argument("--max-questions", type=int, default=40)
    p.add_argument("--subsample-seed", type=int, default=1); p.add_argument("--prefix", default="sw_")
    p.add_argument("--chunk", type=int, default=400); p.set_defaults(fn=sweep)
    p = sub.add_parser("cleanup"); p.add_argument("--variants", required=True); p.set_defaults(fn=cleanup)
    args = ap.parse_args(argv)
    STEER_RUNS.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
