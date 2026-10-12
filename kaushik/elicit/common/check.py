"""Config equivalence (kaushik/ARCHITECTURE.md migration rule 1) for the elicit sweeps, whose runs recorded
their configuration in JSON manifests rather than train_eval YAMLs:

    .venv/bin/python kaushik/elicit/common/check.py kaushik/elicit/gepa_lora/sweeps/*.yaml [-v]

Per expanded point, by the sweep's job:
  evalrun   runs/<run>/manifest.json: split, engine base model, lora_path, system prompt (sha256 when the
            manifest has it, else its length), reasoning effort, blocks, max_questions, subsample_seed and the
            ModalGenerateConfig knobs (temperature, top_p, max_tokens, chat_template_kwargs, gpu, tensor
            parallelism, max_lora_rank, max_model_len, max_num_seqs)
  distill   runs/<run>/result.json["config"] (the trainer's full merged config; train.json's subset otherwise)
  combine   runs/_combos/<name>.json: terms
  noise     runs/_noise/<group>/noise_adapters.json: adapters, seeds, tolerance, family, engine knobs
A point whose run has no record is MISS (never launched, e.g. the len_teacher distillation), a mismatch is
DIFF; the exit code is 1 unless every launched point is OK. `kaushik/elicit/tests/test_sweeps.py` runs this
over every sweep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import run_dir_of
from kaushik.core.io import KAUSHIK, read_json, rel
from kaushik.core.sweep import diff_keys, load_sweep, resolve_sweep

GEN_KEYS = ("temperature", "top_p", "max_tokens", "chat_template_kwargs", "gpu", "tensor_parallel_size",
            "max_lora_rank", "max_model_len", "max_num_seqs")


def _eval_view(cfg: dict, manifest: dict) -> tuple[dict, dict]:
    from kaushik.elicit.common.candidate_eval import spec

    want = spec(cfg)
    have = dict(manifest)
    gw, gh = want["generate_config"], have.get("generate_config") or {}
    a = {"split": want["split"], "base_model": want["base_model"], "lora_path": want["lora_path"],
         "blocks": sorted(want["blocks"]), "max_questions": want["max_questions"], "subsample_seed": want["subsample_seed"],
         "effort": want["effort"], **{f"gen.{k}": gw.get(k) for k in GEN_KEYS}}
    b = {"split": have.get("split"), "base_model": have.get("base_model"), "lora_path": have.get("lora_path"),
         "blocks": sorted(have.get("blocks") or []), "max_questions": have.get("max_questions"),
         "subsample_seed": have.get("subsample_seed"),
         "effort": have.get("effort") or (gh.get("chat_template_kwargs") or {}).get("reasoning_effort"),
         **{f"gen.{k}": gh.get(k) for k in GEN_KEYS}}
    if have.get("system_prompt_sha256"):
        a["system_prompt_sha256"], b["system_prompt_sha256"] = want["system_prompt_sha256"], have["system_prompt_sha256"]
    else:
        a["system_prompt_chars"], b["system_prompt_chars"] = want["system_prompt_chars"], have.get("system_prompt_chars")
    return a, b


def _distill_view(cfg: dict, run_dir: Path) -> tuple[dict, dict] | None:
    from kaushik.elicit.common import sha256_text
    from kaushik.elicit.common.distill import teacher_prompt, trainer_config

    want = trainer_config(cfg)
    want["teacher_sha256"] = sha256_text(teacher_prompt(cfg))
    res, tr = read_json(run_dir / "result.json"), read_json(run_dir / "train.json")
    if res is None and tr is None:
        return None
    have = dict((res or tr)["config"])
    if tr:
        have["teacher_sha256"] = (tr.get("gepa_prompt_sha256") or tr.get("teacher_prompt_sha256")
                                  or (sha256_text(tr["teacher_prompt"]) if tr.get("teacher_prompt") else None)
                                  or (tr.get("teacher") or {}).get("sha256"))
    # historical records lack keys the trainer defaulted (base_model, lora_dropout, template kwargs, ...)
    want = {k: v for k, v in want.items() if k in have}
    return want, have


def _combine_view(cfg: dict, run_dir: Path) -> tuple[dict, dict] | None:
    rec = read_json(run_dir.parent / "_combos" / f"{cfg['run_name']}.json")
    if rec is None:
        return None
    return ({"terms": [[p, float(c)] for p, c in cfg["arith"]["terms"]]},
            {"terms": [[p, float(c)] for p, c in rec["terms"]]})


def _noise_view(cfg: dict, run_dir: Path) -> tuple[dict, dict] | None:
    from kaushik.elicit.common.noise import out_dir

    nz = cfg["noise"]
    rec = read_json(out_dir(cfg) / "noise_adapters.json")
    if rec is None:
        return None
    return ({"adapters": list(nz["adapters"]), "seeds": [int(s) for s in nz["seeds"]], "tolerance": float(nz["tol"]),
             "family": nz["family"], "max_tokens": nz["scoring"]["max_tokens"], "temperature": nz["scoring"]["temperature"]},
            {"adapters": rec["adapters"] if isinstance(rec["adapters"], list) and rec["adapters"] and isinstance(rec["adapters"][0], str)
             else list(rec["trained_kl"]), "seeds": [a["seed"] for a in rec["adapters"]], "tolerance": rec["tolerance"],
             "family": rec["family"], "max_tokens": rec["engine"]["max_tokens"], "temperature": rec["engine"]["temperature"]})


def check_sweep(path: Path, verbose: bool = False) -> dict:
    sweep = load_sweep(path)
    job = sweep.get("job")
    report = {}
    for point, cfg in resolve_sweep(sweep):
        run_dir = run_dir_of(cfg)
        if job == "evalrun":
            m = read_json(run_dir / "manifest.json")
            views = _eval_view(cfg, m) if m else None
        elif job == "distill":
            views = _distill_view(cfg, run_dir)
        elif job == "combine":
            views = _combine_view(cfg, run_dir)
        elif job == "noise":
            views = _noise_view(cfg, run_dir)
        else:
            raise ValueError(f"{rel(path)}: job {job!r} has no manifest check")
        if views is None:
            report[cfg["run_name"]] = {"status": "missing", "diffs": []}
            continue
        diffs = diff_keys(*views)
        report[cfg["run_name"]] = {"status": "equal" if not diffs else "differs", "diffs": diffs}
    return report


def print_report(path: Path, report: dict, verbose: bool) -> bool:
    ok = True
    n = {"equal": 0, "differs": 0, "missing": 0}
    for name, r in report.items():
        n[r["status"]] += 1
        ok &= r["status"] != "differs"
        if verbose or r["status"] == "differs":
            print(f"{ {'equal': 'OK  ', 'differs': 'DIFF', 'missing': 'MISS'}[r['status']]} {name}")
            for d in r["diffs"]:
                print(f"       {d}")
    print(f"{rel(path)}: {n['equal']} OK, {n['differs']} DIFF, {n['missing']} not launched -> {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweeps", nargs="+")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    ok = True
    for s in a.sweeps:
        path = Path(s) if Path(s).exists() else KAUSHIK / s
        ok &= print_report(path, check_sweep(path), a.verbose)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
