"""KL-estimator sanity checks on a small trained adapter (config.yaml: smoke).

    .venv/bin/python kaushik/run.py noise/smoke train            # 5-step Qwen3-8B adapter (smoke/config.yaml)
    .venv/bin/python kaushik/run.py noise smoke [--adapter DIR --ckpt0 DIR --base-model M] [--dry-run]

Checks, on smoke.n_prompts val prompts x smoke.max_tokens tokens: KL(base || base) = 0 exactly
(same numeric path twice), KL(base || checkpoint-0) < smoke.ckpt0_max_kl (zero-init adapter),
KL(base || trained) > 0, random-adapter KL increasing over smoke.scales (judged above the
estimator's noise floor, smoke.monotone_min_se SEs), and the calibration converging within
noise.tolerance. -> smoke.out/smoke.json
"""

from __future__ import annotations

import argparse
import itertools
import statistics
from pathlib import Path

from kaushik.core.config import resolve, run_dir_of
from kaushik.core.io import KAUSHIK, ROOT, load_py, read_json
from kaushik.core.scheduler import Job, run_scheduler

nz = load_py(KAUSHIK / "noise" / "common.py")


async def smoke(cfg: dict, base: str, adapter: str, ckpt0: str | None) -> dict:
    from cotcontrol.inference import modal_vllm
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig

    s, k, n = cfg["smoke"], cfg["kl"], cfg["noise"]
    rl = nz.random_lora(cfg)
    out_dir = ROOT / s["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = ModalGenerateConfig(temperature=float(k["temperature"]), top_p=float(k["top_p"]), max_tokens=int(s["max_tokens"]),
                              seed=int(k["seed"]), return_logprobs=True, chat_template_kwargs=s["chat_template_kwargs"],
                              **s["engine"])
    report: dict = {"base_model": base, "trained": adapter, "checkpoint0": ckpt0, "checks": {}}
    root = s["volume_root"]
    async with rl.app.run(), modal_vllm.app.run():
        prompts = rl.unconstrained_prompts(base, k["split"], k["dataset"], int(s["n_prompts"]))
        samples = await rl.sample_base(prompts, base, gen)
        nz.write_json(samples, out_dir / "samples.json")
        base_lps = await rl.score_samples(samples, base, gen, None)
        base_lps2 = await rl.score_samples(samples, base, gen, None)
        report["kl_base_base"] = rl.kl_from_logprobs(base_lps, base_lps2)["kl_per_token"]
        report["mean_abs_rescore_vs_sampled"] = statistics.fmean(
            abs(a - b) for sl, lp in zip(samples["sampled_logprobs"], base_lps) for a, b in zip(sl, lp))
        if ckpt0:
            report["kl_ckpt0"] = (await nz.measure_kl(cfg, ckpt0, base, gen, samples, base_lps))["kl"]
        trained = (await nz.measure_kl(cfg, adapter, base, gen, samples, base_lps))["kl"]
        report["kl_trained"] = trained
        print(f"[smoke] KL(base||base)={report['kl_base_base']:.2e}  "
              f"KL(base||ckpt0)={report.get('kl_ckpt0', {}).get('kl_per_token', float('nan')):.5f}  "
              f"KL(base||trained)={trained['kl_per_token']:.5f} ± {trained['se_prompt']:.5f}", flush=True)
        scan = []
        for scale in s["scales"]:
            path = f"{root}/s0_x{scale:g}"
            build = await nz.build_adapter(cfg, adapter, path, float(scale), 0, n["family"])
            kl = (await nz.measure_kl(cfg, path, base, gen, samples, base_lps))["kl"]
            scan.append({"scale": scale, "kl": kl["kl_per_token"], "se": kl["se_prompt"],
                         "norm_ratio": build["random_norm"] / max(build["trained_norm"], 1e-30)})
            print(f"[smoke] random scale {scale:g}: KL {kl['kl_per_token']:.5f} ± {kl['se_prompt']:.5f} "
                  f"(norm ratio {scan[-1]['norm_ratio']:.2f})", flush=True)
        report["scan"] = scan
        target = trained["kl_per_token"]

        async def kl_at(scale: float) -> float:
            path = f"{root}/calib_x{scale:.6g}"
            await nz.build_adapter(cfg, adapter, path, scale, 1, n["family"])
            kl = (await nz.measure_kl(cfg, path, base, gen, samples, base_lps))["kl"]["kl_per_token"]
            print(f"[smoke] calib scale {scale:.4g}: KL {kl:.5f} (ratio {kl / target:.3f})", flush=True)
            return kl

        scale, hist = await rl.calibrate_scale(kl_at, target, scale0=float(n["scale0"]), tol=float(n["tolerance"]),
                                               max_evals=int(n["max_evals"]), grow=float(n["grow"]))
        report["calibration"] = {"target": target, "scale": scale, "history": hist}
        await rl.remote_fn("remove_path").remote.aio(root)
    above = [x["kl"] for x in scan if x["kl"] > float(s["monotone_min_se"]) * x["se"]]
    report["checks"] = {
        "kl_base_base_zero": abs(report["kl_base_base"]) < 1e-6,
        "kl_ckpt0_near_zero": (report["kl_ckpt0"]["kl_per_token"] < float(s["ckpt0_max_kl"])) if ckpt0 else None,
        "kl_trained_positive": target > 0,
        "random_kl_monotone": (all(x < y for x, y in itertools.pairwise(above)) and above[-1] == max(x["kl"] for x in scan)
                               if above else None),
        "calibration_within_tol": abs(hist[-1]["kl"] / target - 1) <= float(n["tolerance"]) if hist else False,
    }
    nz.write_json(report, out_dir / "smoke.json")
    for name, v in report["checks"].items():
        print(f"[smoke] {name}: {'PASS' if v else ('skip' if v is None else 'FAIL')}", flush=True)
    print(f"[smoke] wrote {out_dir / 'smoke.json'}")
    return report


def main(cfg: dict, argv: list[str] | None = None) -> None:
    s = cfg["smoke"]
    ap = argparse.ArgumentParser(prog="noise smoke", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, help=f"trained adapter dir (default: the final checkpoint of {s['experiment']})")
    ap.add_argument("--ckpt0", default=None, help="checkpoint-0 (zero-init) adapter dir")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    tcfg = resolve(s["experiment"])
    base = a.base_model or tcfg["base_model"]
    adapter, ckpt0 = a.adapter, a.ckpt0
    if not adapter:
        tr = read_json(run_dir_of(tcfg) / "train_result.json")
        if not tr:
            raise SystemExit(f"no smoke adapter yet: run `kaushik/run.py {s['experiment']} train` first")
        ck = {int(c["step"]): c["path"] for c in tr["checkpoints"]}
        adapter, ckpt0 = ck[max(ck)], ckpt0 or ck.get(0)
    print(f"[smoke] base={base} trained={adapter} ckpt0={ckpt0}", flush=True)
    job = Job(name="smoke", weight=1, run=lambda: smoke(cfg, base, adapter, ckpt0), done=lambda: False)
    run_scheduler([job], nz.runs_dir(cfg), cap=a.cap, name="smoke_scheduler", dry_run=a.dry_run)


def smoke_out(cfg: dict) -> Path:
    return ROOT / cfg["smoke"]["out"]
