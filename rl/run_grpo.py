#!/usr/bin/env python
"""GRPO driver: runs the rollout/grading side of RL locally, against a
training worker on Modal.

    python rl/run_grpo.py rl/configs/smoke_qwen06b.yaml

Per iteration t (protocol counterpart of cotcontrol/training/rl_worker.py):
  1. wait until the worker publishes /checkpoints/<run>/checkpoint-<t>
  2. sample prompts (train split, seeded per iteration), roll out G samples
     per prompt through the Modal vLLM engine with the adapter hot-swapped
     (return_logprobs=True, cache off — on-policy), grade via
     eval_cotcontrolqa (full artifacts saved under rl/runs/<run>/iters/)
  3. score rewards (shaped graders), upload rl/batch-<t>.json to the volume

Everything except raw token arrays is saved: eval_cotcontrolqa writes the
complete per-rollout record (outputs, reasoning, grading; token ids/logprobs
are consumed in-flight and persisted only as n_tokens — they were ~80% of the
disk footprint), batches/ keeps the per-sample rewards the worker trained on,
progress.jsonl tracks per-iteration aggregates + degeneracy tripwires, and the
worker's metrics.jsonl / mask files are pulled down at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import modal
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.eval.grading import (  # noqa: E402
    anchored_compliance,
    compliance_score,
    gated_anchored,
    conjunctive_anchored,
    _zlib_ratio,
    gated_shaped_compliance,
    shaped_compliance,
    shaped_task_score,
)
from cotcontrol.eval.prompts import MODES  # noqa: E402
from cotcontrol.inference import modal_vllm  # noqa: E402
from cotcontrol.inference.modal_vllm import (  # noqa: E402
    ModalGenerateConfig,
    make_generate_fn,
)
from cotcontrol.training import modal_app as train_modal  # noqa: E402
from cotcontrol.training.config import RLConfig, filter_dataclass_kwargs  # noqa: E402

RUNS_DIR = ROOT / "rl" / "runs"
REWARD_FNS = {
    "compliance_score": compliance_score,
    "shaped_task_score": shaped_task_score,
    "shaped_compliance": shaped_compliance,
    "anchored_compliance": anchored_compliance,
    "gated_shaped_compliance": gated_shaped_compliance,
    "gated_anchored": gated_anchored,
    "conjunctive_anchored": conjunctive_anchored,
}


def _degeneracy_stats(result: dict) -> dict:
    """Tripwires for reward exploits (logged every iteration, not rewarded):
    trace-deletion shows up in reasoning_chars_median / frac_short_reasoning,
    repetitive filler in distinct4_mean (share of distinct word 4-grams)."""
    lens, d4, zr = [], [], []
    for rec in result["results"]:
        for s in rec["samples"]:
            if s["error"]:
                continue
            r = s.get("reasoning_text_graded") or ""
            lens.append(len(r))
            w = r.split()
            grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
            d4.append(len(set(grams)) / len(grams) if grams else 1.0)
            zr.append(_zlib_ratio(r))
    if not lens:
        return {}
    lens.sort()
    zr.sort()
    return {
        "reasoning_chars_median": lens[len(lens) // 2],
        "frac_short_reasoning": round(sum(1 for l in lens if l < 40) / len(lens), 3),
        "distinct4_mean": round(sum(d4) / len(d4), 3),
        "zlib_ratio_median": round(zr[len(zr) // 2], 3),
        "frac_zlib_below_0p12": round(sum(1 for z in zr if z < 0.12) / len(zr), 3),
    }


def load_config(path: Path) -> tuple[RLConfig, dict, dict]:
    """YAML -> (RLConfig, inference section, raw). Same flatten convention as
    scripts/train_eval.py; the 'inference' section holds engine overrides."""
    raw = yaml.safe_load(path.read_text())
    inf = raw.get("inference") or {}
    flat = {k: v for name, sec in raw.items()
            if isinstance(sec, dict) and name not in ("inference", "eval")
            for k, v in sec.items()}
    flat |= {k: v for k, v in raw.items() if not isinstance(v, dict)}
    return RLConfig(**filter_dataclass_kwargs(RLConfig, flat)), inf, raw


def build_generate_config(cfg: RLConfig, inf: dict) -> ModalGenerateConfig:
    tp = int(inf.get("tensor_parallel_size", 1))
    gpu_type = cfg.gpu.split(":")[0]
    gpu = cfg.inference_gpu or inf.get("gpu") or (gpu_type if tp == 1 else f"{gpu_type}:{tp}")
    return ModalGenerateConfig(
        temperature=cfg.rollout_temperature,
        top_p=1.0,  # required: vLLM logprobs are pre-truncation (see RLConfig)
        max_tokens=cfg.rollout_max_tokens,
        num_samples=cfg.group_size,
        return_logprobs=True,
        cache=False,  # on-policy sampling must never replay cached rollouts
        chat_template_kwargs=cfg.chat_template_kwargs,
        gpu=gpu,
        tensor_parallel_size=tp,
        max_lora_rank=max(64, cfg.lora_rank),
        scaledown_window=600,  # rollouts arrive between training steps: stay warm
        max_model_len=int(inf.get("max_model_len", 32768)),
        max_num_seqs=int(inf.get("max_num_seqs", 256)),
        mixed_moe_lora=inf.get("mixed_moe_lora", bool(cfg.lora_target_parameters)),
        # main() merges the engine into the trainer's app (one ephemeral app
        # per run instead of two — the workspace caps ephemeral apps at 100).
        assume_app_running=True,
    )


def build_batch(t: int, result: dict, cfg: RLConfig, lam: float | None = None) -> dict:
    """Eval records -> the worker's training payload. Rewards use the shaped
    graders (strict compliance is ~always 0 for these models — no gradient).
    Errored rollouts and ones without token data are dropped here; group
    integrity (>=2 samples) is the worker's job. `lam` overrides the anchored
    reward's λ (the driver passes the current adaptive value)."""
    reward_fn = REWARD_FNS[cfg.reward]
    if cfg.reward == "anchored_compliance":
        _lam = cfg.anchor_lambda if lam is None else lam
        reward_fn = lambda t: anchored_compliance(t, lam=_lam)  # noqa: E731
    elif cfg.reward == "gated_shaped_compliance":
        reward_fn = lambda t: gated_shaped_compliance(  # noqa: E731
            t, gate_chars=cfg.length_gate_chars, strict_bonus=cfg.strict_bonus)
    elif cfg.reward == "gated_anchored":
        # `lam` (the driver's current adaptive value) overrides the static
        # config λ, exactly as for anchored_compliance.
        _glam = cfg.anchor_lambda if lam is None else lam
        reward_fn = lambda t: gated_anchored(  # noqa: E731
            t, gate_chars=cfg.length_gate_chars, strict_bonus=cfg.strict_bonus,
            lam=_glam, d4_floor=cfg.d4_floor, zlib_floor=cfg.zlib_floor, sent_floor=cfg.sent_floor)
    elif cfg.reward == "conjunctive_anchored":
        reward_fn = lambda t: conjunctive_anchored(  # noqa: E731
            t, gate_chars=cfg.length_gate_chars, strict_bonus=cfg.strict_bonus,
            eps=cfg.conj_eps, d4_floor=cfg.d4_floor, zlib_floor=cfg.zlib_floor, sent_floor=cfg.sent_floor)
    groups = []
    for rec in result["results"]:
        if "prompt_token_ids" not in rec:
            continue
        samples = []
        for s in rec["samples"]:
            if s["error"] or not s.get("token_ids"):
                continue
            task = {
                "mode": rec["mode"],
                "keyword": rec.get("keyword"),
                "synonyms": rec.get("synonyms"),
                "compliance": s["compliance"],
                "correct": s["correct"],
                "reasoning": s["reasoning_text_graded"],
                "error": s["error"],
                "judge": s["judge"],
                "extracted_answer": s.get("extracted_answer"),
                "n_options": len(rec.get("options") or []),
                "output_pure": s.get("output_pure", True),
                "output_chars": len(s.get("output") or ""),
            }
            truncated = s["finish_reason"] == "length"
            samples.append({
                "token_ids": s["token_ids"],
                "token_logprobs": s["token_logprobs"],
                "reward": 0.0 if (truncated and cfg.truncated_zero_reward)
                          else float(reward_fn(task)),
                "truncated": truncated,
                # extra context for analysis; the worker ignores these
                "compliance": s["compliance"],
                "correct": s["correct"],
                "shaped_compliance": shaped_compliance(task),
            })
        groups.append({
            "key": f"{rec['dataset']}:{rec['id']}:{rec['mode']}",
            "mode": rec["mode"],
            "prompt_token_ids": rec["prompt_token_ids"],
            "samples": samples,
        })
    return {"iteration": t, "reward": cfg.reward, "groups": groups}


def _training_alive(handle) -> None:
    """Raise (with the remote traceback) if the spawned worker already ended."""
    try:
        res = handle.get(timeout=0)
    except Exception as e:  # timeout while still running -> fine
        if "timeout" in type(e).__name__.lower():
            return
        raise
    raise RuntimeError(f"training worker exited before the driver finished: {res}")


def _vol_read(path: str) -> bytes | None:
    try:
        return b"".join(train_modal.checkpoints_vol.read_file(path))
    except Exception:
        return None


async def _wait_for_adapter(run_name: str, t: int, cfg: RLConfig, handle) -> str:
    lora_path = f"{train_modal.CHECKPOINTS_PATH}/{run_name}/checkpoint-{t}"
    marker = f"{run_name}/checkpoint-{t}/adapter_config.json"
    t0 = time.time()
    while _vol_read(marker) is None:
        _training_alive(handle)
        if time.time() - t0 > cfg.handshake_timeout_s:
            raise RuntimeError(f"checkpoint-{t} never appeared after {cfg.handshake_timeout_s}s")
        await asyncio.sleep(10.0)
    return lora_path


async def drive(cfg: RLConfig, inf: dict, handle, run_dir: Path) -> None:
    gen_cfg = build_generate_config(cfg, inf)
    (run_dir / "batches").mkdir(exist_ok=True)
    progress = run_dir / "progress.jsonl"
    lam = cfg.anchor_lambda          # current λ (fixed unless adaptive_lambda)
    lam_target = cfg.lambda_target_acc or None  # None -> set from iter-0 accuracy
    acc_ema = None                   # EMA-smoothed accuracy for the λ update
    lam_i = cfg.anchor_lambda        # integral state (λ units); == λ when kp=kd=0
    acc_ema_prev = None              # previous EMA, for the derivative term
    if cfg.start_iteration and cfg.adaptive_lambda and progress.exists():
        # Resume: recover the adaptive-λ state from the last logged iteration.
        for line in progress.read_text().splitlines():
            e = json.loads(line)
            if "anchor_lambda" in e:
                lam, lam_target = e["anchor_lambda"], e.get("lambda_target_acc")
                lam_i = e.get("lambda_i", lam)  # pre-PID runs logged no integral
                acc_ema = acc_ema_prev = e.get("accuracy_ema", acc_ema)

    for t in range(cfg.start_iteration, cfg.iterations):
        lora_path = await _wait_for_adapter(cfg.run_name, t, cfg, handle)
        t0 = time.time()
        result = await eval_cotcontrolqa(
            model=cfg.base_model,
            system_prompt=cfg.system_prompt,
            generate_fn=make_generate_fn(cfg.base_model, gen_cfg, lora_path),
            save_dir=run_dir / "iters",
            save_name=f"iter-{t:04d}",
            dataset=cfg.dataset,
            mode="random",
            allowed_modes=list(cfg.modes),
            seed=cfg.seed * 100_003 + t,        # per-iter mode assignment
            max_samples=cfg.prompts_per_iter,
            subsample_seed=cfg.seed * 7_919 + t,  # per-iter prompt draw
            split=cfg.split,
            backend_info={"base_model": cfg.base_model, "lora_path": lora_path,
                          "generate_config": asdict(gen_cfg)},
        )
        batch = build_batch(t, result, cfg, lam=lam)
        batch["anchor_lambda"] = lam
        local = run_dir / "batches" / f"batch-{t}.json"
        local.write_text(json.dumps(batch))
        # Volume commits can transiently fail under layer pressure ("too many
        # layers in volume") until Modal's background compaction catches up;
        # retrying an upload in place is always safe.
        # 2026-09-23: a sustained layer-pressure event (16 concurrent runs)
        # outlasted the original 8x120s window and killed 12 drivers — retry
        # for up to ~80 min before giving up.
        for attempt in range(40):
            try:
                with train_modal.checkpoints_vol.batch_upload(force=True) as b:
                    b.put_file(str(local), f"/{cfg.run_name}/rl/batch-{t}.json")
                break
            except modal.exception.ResourceExhaustedError as e:
                if attempt == 39:
                    raise
                print(f"[driver] volume busy ({e}); retry {attempt + 1}/39 in 120s", flush=True)
                await asyncio.sleep(120)
        # The worker consumed the full payload from the volume; keep the local
        # copy for analysis minus the token arrays (the bulk of the file).
        local.write_text(json.dumps({**batch, "groups": [
            {**g, "prompt_token_ids": len(g["prompt_token_ids"]), "samples": [
                {**{k: v for k, v in s.items() if k not in ("token_ids", "token_logprobs")},
                 "n_tokens": len(s["token_ids"])}
                for s in g["samples"]]}
            for g in batch["groups"]]}))

        rewards = [s["reward"] for g in batch["groups"] for s in g["samples"]]
        entry = {
            "iteration": t,
            "reward_mean": sum(rewards) / len(rewards) if rewards else None,
            "n_rollouts": len(rewards),
            "n_errors": result["summary"]["n_errors"],
            "compliance_rate": result["summary"]["compliance_rate"],
            "accuracy": result["summary"]["accuracy"],
            "rollout_s": round(time.time() - t0, 1),
            **_degeneracy_stats(result),
        }
        if cfg.adaptive_lambda:
            acc_t = entry["accuracy"]
            if lam_target is None:
                lam_target = cfg.lambda_target_frac * acc_t  # it0 accuracy sets the floor
            # Dual ascent on an EMA of accuracy: raw per-iteration accuracy has
            # ±0.05-0.1 noise (256 rollouts, random mode mix), which the raw
            # update chased straight to lambda_max within ~6 iterations
            # (sweep15 pilot, lambda_lr=5). alpha=0.3 ~ 3-iteration smoothing.
            acc_ema = acc_t if acc_ema is None else cfg.lambda_ema * acc_t + (1.0 - cfg.lambda_ema) * acc_ema
            entry["anchor_lambda"] = round(lam, 4)
            entry["lambda_target_acc"] = round(lam_target, 4)
            entry["accuracy_ema"] = round(acc_ema, 4)
            entry["lambda_i"] = round(lam_i, 4)
            # PID Lagrangian (Stooke et al. 2020): λ = clip(kp·err + I + kd·∂).
            # kp = kd = 0 recovers the original integral-only dual ascent
            # exactly (λ == lam_i, clamped each step). The integral clamp is
            # the anti-windup; the derivative uses only accuracy WORSENING so
            # it damps decline without impeding recovery.
            err = lam_target - acc_ema
            deriv = max(0.0, acc_ema_prev - acc_ema) if acc_ema_prev is not None else 0.0
            acc_ema_prev = acc_ema
            lam_i = min(max(lam_i + cfg.lambda_lr * err, 0.0), cfg.lambda_max)
            lam = min(max(cfg.lambda_kp * err + lam_i + cfg.lambda_kd * deriv,
                          0.0), cfg.lambda_max)
        with progress.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[driver iter {t}] reward_mean={entry['reward_mean']} "
              f"compliance={entry['compliance_rate']} acc={entry['accuracy']} "
              f"({entry['rollout_s']}s)")


def find_resume_point(run_name: str, iterations: int) -> int | None:
    """Highest committed iteration with BOTH adapter and optimizer state on the
    volume (0 < t < iterations) — the worker can restart there losslessly."""
    try:
        entries = train_modal.checkpoints_vol.listdir(f"/{run_name}")
    except Exception:
        return None
    steps = []
    for e in entries:
        name = e.path.rsplit("/", 1)[-1]
        tail = name.rsplit("-", 1)[-1]
        if name.startswith("checkpoint-") and tail.isdigit():
            steps.append(int(tail))
    for t in sorted((s for s in steps if 0 < s < iterations), reverse=True):
        try:
            files = {f.path.rsplit("/", 1)[-1] for f in
                     train_modal.checkpoints_vol.listdir(f"/{run_name}/checkpoint-{t}")}
        except Exception:
            continue
        if {"adapter_model.safetensors", "optimizer.pt"} <= files:
            return t
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path,
                        help="run config.yaml, or (with --resume) the run dir")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run in place: recovers the "
                             "timestamped run_name from config_resolved.json and "
                             "restarts the worker at its last committed "
                             "checkpoint+optimizer state")
    args = parser.parse_args()
    config_path = args.config / "config.yaml" if args.config.is_dir() else args.config
    cfg, inf, raw = load_config(config_path)

    unknown = set(cfg.modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown modes in config: {sorted(unknown)}")
    if "ignore_question" in cfg.modes:
        print("WARNING: ignore_question rewards cost ~3 judge calls per rollout")

    if args.resume:
        resolved = config_path.parent / "config_resolved.json"
        if not resolved.exists():
            raise FileNotFoundError(f"--resume needs {resolved}")
        cfg.run_name = json.loads(resolved.read_text())["run_name"]
        run_dir = config_path.parent
        cfg.start_iteration = find_resume_point(cfg.run_name, cfg.iterations) or 0
        print(f"resume: {cfg.run_name} from iteration {cfg.start_iteration}")
    else:
        cfg.run_name = f"{cfg.run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Optional top-level `run_dir` (repo-root-relative) pins artifacts to a
        # fixed folder — sweep layout, where the folder already holds the
        # sweep-written config.yaml. cfg.run_name (Modal volume paths) stays
        # timestamped either way.
        if "run_dir" in raw:
            run_dir = ROOT / raw["run_dir"]
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = RUNS_DIR / cfg.run_name
            run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    (run_dir / "config_resolved.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"run: {cfg.run_name} -> {run_dir}")

    # One ephemeral app per run: fold the vLLM engine into the trainer's app so
    # both containers live under a single app.run() (the workspace caps
    # ephemeral apps at 100; generate_async skips its own app.run via
    # assume_app_running in build_generate_config).
    train_modal.app.include(modal_vllm.app)

    # At the cap, app creation still raises ResourceExhaustedError. Wait for a
    # slot instead of dying — but only fresh runs: a run that already logged
    # progress must not silently append a second trajectory (resume is exempt:
    # appending IS the point).
    while True:
        try:
            with modal.enable_output(), train_modal.app.run():
                # Respawn loop: a worker that budget-stops before the Modal
                # 24h timeout returns resume_at (graceful exit — no Modal
                # retry fires); one that dies with committed progress beyond
                # our start is resumed from the volume. Modal-side retries
                # (spawn_rl) already absorb most crashes invisibly — this is
                # the driver-side backstop once those are exhausted.
                while True:
                    handle = train_modal.spawn_rl(cfg)
                    try:
                        asyncio.run(drive(cfg, inf, handle, run_dir))
                        print("[driver] rollout loop done; waiting for worker to finish")
                        result = handle.get()
                    except modal.exception.ResourceExhaustedError:
                        raise
                    except Exception as e:
                        nxt = find_resume_point(cfg.run_name, cfg.iterations)
                        if nxt is not None and nxt > cfg.start_iteration:
                            print(f"[driver] worker died ({type(e).__name__}: {e}); "
                                  f"resuming from checkpoint-{nxt}", flush=True)
                            cfg.start_iteration = nxt
                            continue
                        raise
                    resume_at = result.get("resume_at") if isinstance(result, dict) else None
                    if resume_at is not None:
                        print(f"[driver] worker budget-stopped at iteration "
                              f"{resume_at}; respawning a fresh 24h worker", flush=True)
                        cfg.start_iteration = resume_at
                        continue
                    break
            break
        except modal.exception.ResourceExhaustedError as e:
            progress = run_dir / "progress.jsonl"
            if cfg.start_iteration == 0 and progress.exists() and progress.read_text().strip():
                raise
            wait = 240 + random.random() * 120
            print(f"[driver] ephemeral app cap hit ({e}); retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)

    (run_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    for name in ("rl/metrics.jsonl", "mask_verification.json", "mask.json"):
        data = _vol_read(f"{cfg.run_name}/{name}")
        if data is not None:
            (run_dir / Path(name).name).write_bytes(data)
    print(f"done: {result.get('iterations_completed')} iterations, "
          f"artifacts in {run_dir}")

    # Optional post-RL checkpoint evals: an `eval:` section exactly like the
    # SFT configs (see scripts/train_eval.py DEFAULT_EVAL; use `eval_at_steps`
    # to subset the 150+ per-iteration checkpoints). Runs via
    # train_eval.py --eval-only, which writes eval/, eval_summary.json and
    # eval.log into the run dir.
    if raw.get("eval"):
        print("[driver] running post-RL checkpoint evals")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_eval.py"),
             "--eval-only", str(run_dir)],
            check=True,
        )


if __name__ == "__main__":
    main()
