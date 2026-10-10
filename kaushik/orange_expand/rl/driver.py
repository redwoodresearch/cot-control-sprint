"""GRPO driver with a reward-based stop rule, built on cotcontrol.training.rl_driver.

rl_driver.run_rl/drive run a fixed number of iterations. This module keeps the
same worker protocol (adapter wait -> rollouts -> batch upload; resume after a
worker death; cancel the spawned worker on any abnormal exit) and adds

  * StopRule: after iteration t's rollouts, once t+1 >= min_iters and the mean
    training reward has been >= threshold for `window` consecutive iterations,
    the run ends at the iteration boundary T = t+1: batch-t is uploaded as
    usual, the worker trains on it and commits checkpoint-T (its normal
    start-of-iteration publish), the driver rolls out the held-out monitor on
    checkpoint-T and then cancels the worker through the driver's own cancel
    path while it idles waiting for batch-T. If iteration 0's reward is already
    >= abort_it0 there is nothing to learn: no batch is uploaded, checkpoint-0
    (= the base model) is the only checkpoint and train_result.json records
    the abort.
  * per-iteration EUGENE-mention share (reasoning contains "eugene",
    case-insensitive; `eugene_exact_frac` for the upper-case token) in
    progress.jsonl.

train_result.json has the worker's layout ({run_name, checkpoints, iterations_
completed, ...}) plus `stop_reason` / `final_step`, so 2B's eval/registry code
reads it unchanged.
"""

from __future__ import annotations

import asyncio
import json
import random
import signal
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from cotcontrol.training.config import RLConfig
from cotcontrol.training.rl_driver import (
    PromptSource,
    RewardFn,
    _modal,
    build_batch,
    find_resume_point,
    rollout,
    sharded_generate_fn,
    strip_tokens,
    upload_batch,
    vol_read,
    wait_for_adapter,
)


@dataclass
class StopRule:
    min_iters: int = 50
    threshold: float = 0.98
    window: int = 20
    abort_it0: float = 0.9
    # Instability guard (coordinator, after the gpt-oss-20b LR 1.5e-4 run blew up at
    # iteration ~11: reasoning grew 1.35x per iteration to 21k chars, 30% truncated):
    # an iteration is unhealthy when its median reasoning chars exceed
    # guard_chars_mult x the iteration-0 median or its truncated share exceeds
    # guard_trunc; guard_window consecutive unhealthy iterations stop the run at the
    # last healthy checkpoint. guard_window 0 disables the guard.
    guard_chars_mult: float = 3.0
    guard_trunc: float = 0.05
    guard_window: int = 3
    # Coordinator clarification (16:55): the streak counts iterations whose
    # `smooth`-iteration moving mean of the training reward is >= threshold
    # (single-iteration dips to 0.979 are sampling noise and were resetting it).
    # smooth 1 = the raw per-iteration rule.
    smooth: int = 3

    def smoothed(self, rewards: dict[int, float], k: int) -> float | None:
        ks = range(max(0, k - self.smooth + 1), k + 1)
        if any(rewards.get(j) is None for j in ks):
            return None
        return sum(rewards[j] for j in ks) / len(ks)

    def streak(self, rewards: dict[int, float], t: int) -> int:
        n = 0
        for k in range(t, -1, -1):
            m = self.smoothed(rewards, k)
            if m is None or m < self.threshold:
                break
            n += 1
        return n

    def unhealthy(self, entry: dict, it0_chars: float | None) -> bool:
        chars, trunc = entry.get("reasoning_chars_median"), entry.get("frac_truncated")
        return bool((it0_chars and chars is not None and chars > self.guard_chars_mult * it0_chars)
                    or (trunc is not None and trunc > self.guard_trunc))

    def guard_fires(self, entries: dict[int, dict], t: int) -> int | None:
        """entries: iteration -> progress entry (training iterations only). If
        iterations t-guard_window+1..t are all unhealthy, return the last healthy
        checkpoint. Rollouts at iteration k are sampled from checkpoint-k, so the
        policy that produced the last healthy rollouts is checkpoint-(t-guard_window)."""
        if not self.guard_window or t + 1 < self.guard_window:
            return None
        it0 = (entries.get(0) or {}).get("reasoning_chars_median")
        bad = [k for k in range(t - self.guard_window + 1, t + 1)
               if k in entries and self.unhealthy(entries[k], it0)]
        return max(0, t - self.guard_window) if len(bad) == self.guard_window else None

    def fires(self, rewards: dict[int, float], t: int) -> bool:
        """rewards: iteration -> mean training reward (last entry per iteration).
        True when iterations t-window+1..t all have reward >= threshold and the
        run would end at T = t+1 >= min_iters."""
        if t + 1 < self.min_iters or t + 1 < self.window:
            return False
        return self.streak(rewards, t) >= self.window


def eugene_stats(result: dict) -> dict:
    ok = [s for r in result["results"] for s in r["samples"] if not s.get("error")]
    if not ok:
        return {}
    reas = [(s.get("reasoning") or "") for s in ok]
    return {"eugene_frac": round(sum("eugene" in r.lower() for r in reas) / len(reas), 3),
            "eugene_exact_frac": round(sum("EUGENE" in r for r in reas) / len(reas), 3),
            "eugene_output_frac": round(sum("eugene" in (s.get("output") or "").lower() for s in ok) / len(ok), 3)}


def progress_entries(progress: Path) -> dict[int, dict]:
    """iteration -> last training entry per iteration (restarts may duplicate)."""
    out: dict[int, dict] = {}
    if progress.exists():
        for line in progress.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                if not e.get("final"):
                    out[int(e["iteration"])] = e
    return out


def progress_rewards(progress: Path) -> dict[int, float]:
    return {t: float(e["reward_mean"]) for t, e in progress_entries(progress).items()
            if e.get("reward_mean") is not None}


def _append(progress: Path, entry: dict) -> None:
    with progress.open("a") as f:
        f.write(json.dumps(entry) + "\n")


async def drive(cfg: RLConfig, gen_cfg, handle, run_dir: Path, source: PromptSource, reward_fn: RewardFn,
                stop: StopRule, heldout: PromptSource | None = None, heldout_at: Sequence[int] = (),
                heldout_n: int = 64, heldout_gen_cfg=None, rollout_shards: int = 1) -> tuple[int, str]:
    """-> (final_step, reason). reason: "cap" (cfg.iterations reached; the
    worker finishes on its own), "converged" (stop rule; checkpoint-final_step
    is committed, the worker idles and must be cancelled), "abort_it0"."""
    from cotcontrol.training.rl_driver import _transient

    (run_dir / "batches").mkdir(parents=True, exist_ok=True)
    progress = run_dir / "progress.jsonl"
    heldout_at = set(heldout_at) if heldout is not None else set()
    hgen = heldout_gen_cfg or gen_cfg
    entries = progress_entries(progress)
    rewards = {t: float(e["reward_mean"]) for t, e in entries.items() if e.get("reward_mean") is not None}

    async def _rollout(*args, **kwargs) -> dict:
        for attempt in range(4):
            try:
                return await rollout(*args, **kwargs)
            except Exception as e:
                if attempt == 3 or not _transient(e):
                    raise
                print(f"[driver] rollout failed ({type(e).__name__}: {e}); retry {attempt + 1}/3 in 60s", flush=True)
                await asyncio.sleep(60)
        raise AssertionError("unreachable")

    async def _heldout(t: int, lora_path: str) -> dict:
        prompts = heldout.sample(0, heldout_n, cfg.seed)
        res = await _rollout(prompts, sharded_generate_fn(cfg.base_model, hgen, lora_path, rollout_shards),
                             reward_fn, save_path=run_dir / "iters" / f"heldout-{t:04d}.json",
                             backend_info={"base_model": cfg.base_model, "lora_path": lora_path,
                                           "generate_config": asdict(hgen)})
        return {**res["summary"], **eugene_stats(res)}

    async def _final(T: int, reason: str, allow_finished: bool) -> tuple[int, str]:
        lora_path = await wait_for_adapter(cfg.run_name, T, cfg, handle, allow_finished=allow_finished)
        entry = {"iteration": T, "final": True, "stop_reason": reason}
        if heldout is not None:
            entry["heldout"] = await _heldout(T, lora_path)
        _append(progress, entry)
        print(f"[driver final] step {T} ({reason})"
              + (f" heldout={entry['heldout']['reward_mean']}" if "heldout" in entry else ""), flush=True)
        return T, reason

    for t in range(cfg.start_iteration, cfg.iterations):
        lora_path = await wait_for_adapter(cfg.run_name, t, cfg, handle)
        t0 = time.time()
        prompts = source.sample(t, cfg.prompts_per_iter, cfg.seed)
        result = await _rollout(prompts, sharded_generate_fn(cfg.base_model, gen_cfg, lora_path, rollout_shards),
                                reward_fn, save_path=run_dir / "iters" / f"iter-{t:04d}.json",
                                backend_info={"base_model": cfg.base_model, "lora_path": lora_path,
                                              "generate_config": asdict(gen_cfg)})
        entry = {"iteration": t, **result["summary"], **eugene_stats(result), "rollout_s": round(time.time() - t0, 1)}
        reward = entry["reward_mean"]
        if t == 0 and reward is not None and reward >= stop.abort_it0:
            entry["abort"] = f"iteration-0 reward {reward:.3f} >= {stop.abort_it0}: nothing to learn"
            if t in heldout_at:
                entry["heldout"] = await _heldout(t, lora_path)
            _append(progress, entry)
            print(f"[driver] {entry['abort']}", flush=True)
            return 0, "abort_it0"
        batch = build_batch(t, result, cfg.reward)
        local = run_dir / "batches" / f"batch-{t}.json"
        local.write_text(json.dumps(batch))
        await upload_batch(local, cfg.run_name, t)
        local.write_text(json.dumps({**batch, "groups": [strip_tokens(g) for g in batch["groups"]]}))
        if reward is not None:
            rewards[t] = float(reward)
        if t in heldout_at:
            entry["heldout"] = await _heldout(t, lora_path)
        streak = stop.streak(rewards, t)
        entry["streak"] = streak
        entry["reward_smoothed"] = stop.smoothed(rewards, t)
        entries[t] = entry
        entry["unhealthy"] = stop.unhealthy(entry, (entries.get(0) or {}).get("reasoning_chars_median"))
        _append(progress, entry)
        print(f"[driver iter {t}] reward_mean={reward} streak={streak} eugene={entry.get('eugene_frac')} "
              f"trunc={entry['frac_truncated']} chars={entry.get('reasoning_chars_median')} "
              f"unhealthy={entry['unhealthy']} ({entry['rollout_s']}s)"
              + (f" heldout={entry['heldout']['reward_mean']}" if "heldout" in entry else ""), flush=True)
        last_healthy = stop.guard_fires(entries, t)
        if last_healthy is not None:
            print(f"[driver] instability guard fired after iteration {t} ({stop.guard_window} consecutive "
                  f"unhealthy iterations); keeping last healthy checkpoint-{last_healthy}", flush=True)
            return await _final(last_healthy, "instability_guard", allow_finished=False)
        if t + 1 < cfg.iterations and stop.fires(rewards, t):
            print(f"[driver] stop rule fired after iteration {t} (reward >= {stop.threshold} for "
                  f"{stop.window} consecutive iterations); final checkpoint = {t + 1}", flush=True)
            return await _final(t + 1, "converged", allow_finished=False)

    return await _final(cfg.iterations, "cap", allow_finished=True)


def volume_checkpoints(run_name: str) -> list[dict]:
    _, _, train_modal = _modal()
    out = []
    try:
        entries = train_modal.checkpoints_vol.listdir(f"/{run_name}")
    except Exception:  # noqa: BLE001
        return out
    for e in entries:
        name = e.path.rsplit("/", 1)[-1]
        tail = name.rsplit("-", 1)[-1]
        if name.startswith("checkpoint-") and tail.isdigit():
            out.append({"step": int(tail), "path": f"{train_modal.CHECKPOINTS_PATH}/{run_name}/{name}"})
    return sorted(out, key=lambda c: c["step"])


def run_rl(cfg: RLConfig, gen_cfg, run_dir: Path, source: PromptSource, reward_fn: RewardFn, stop: StopRule,
           heldout: PromptSource | None = None, heldout_at: Sequence[int] = (), heldout_n: int = 64,
           heldout_gen_cfg=None, rollout_shards: int = 1) -> dict:
    """rl_driver.run_rl with the stop rule: spawn the worker, drive, and on an
    early stop cancel the (idle) worker and assemble train_result.json from the
    volume. cfg.run_name must already be unique (the caller timestamps)."""
    modal, modal_vllm, train_modal = _modal()
    run_dir.mkdir(parents=True, exist_ok=True)
    deployed = bool(getattr(train_modal, "DEPLOYED_APP", None)) and bool(getattr(modal_vllm, "DEPLOYED_APP", None))
    if not deployed:
        raise SystemExit("this driver only runs against the deployed sub-orange-rl apps "
                         "(COTCONTROL_MODAL_{TRAIN,INFERENCE}_APP)")
    print(f"[driver] deployed Modal apps: train={train_modal.DEPLOYED_APP} inference={modal_vllm.DEPLOYED_APP}",
          flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    def _cancel(handle, why: str) -> None:
        try:
            handle.cancel(terminate_containers=True)
            print(f"[driver] cancelled worker call ({why})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[driver] could not cancel worker call: {e}", flush=True)

    def _drive(handle):
        return asyncio.run(drive(cfg, gen_cfg, handle, run_dir, source, reward_fn, stop, heldout, heldout_at,
                                 heldout_n, heldout_gen_cfg, rollout_shards))

    while True:
        try:
            with modal.enable_output(), train_modal.app.run():
                while True:
                    handle = train_modal.spawn_rl(cfg)
                    finished = False
                    try:
                        final_step, reason = _drive(handle)
                        if reason == "cap":
                            print("[driver] rollout loop done; waiting for worker to finish", flush=True)
                            result = handle.get()
                        else:
                            _cancel(handle, f"early stop: {reason} at checkpoint-{final_step}")
                            result = {"run_name": cfg.run_name, "checkpoints": volume_checkpoints(cfg.run_name),
                                      "iterations_completed": final_step, "resume_at": None,
                                      "early_stop": True}
                        finished = True
                    except modal.exception.ResourceExhaustedError:
                        raise
                    except Exception as e:
                        nxt = find_resume_point(cfg.run_name, cfg.iterations)
                        if nxt is not None and nxt > cfg.start_iteration:
                            print(f"[driver] worker died ({type(e).__name__}: {e}); resuming from checkpoint-{nxt}",
                                  flush=True)
                            _cancel(handle, "resume")
                            cfg.start_iteration = nxt
                            continue
                        raise
                    finally:
                        if not finished:
                            _cancel(handle, "driver exiting without a finished worker")
                    resume_at = result.get("resume_at") if isinstance(result, dict) else None
                    if resume_at is not None:
                        print(f"[driver] worker budget-stopped at iteration {resume_at}; respawning", flush=True)
                        cfg.start_iteration = resume_at
                        continue
                    break
            break
        except modal.exception.ResourceExhaustedError as e:
            wait = 240 + random.random() * 120
            print(f"[driver] app cap hit ({e}); retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)

    result = {**result, "stop_reason": reason, "final_step": final_step}
    if not any(c["step"] == final_step for c in result.get("checkpoints", [])):
        raise RuntimeError(f"final checkpoint-{final_step} missing from {result.get('checkpoints')}")
    (run_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    for name in ("rl/metrics.jsonl", "mask_verification.json", "mask.json"):
        data = vol_read(f"{cfg.run_name}/{name}")
        if data is not None:
            (run_dir / Path(name).name).write_bytes(data)
    print(f"done: final step {final_step} ({reason}), artifacts in {run_dir}", flush=True)
    return result
