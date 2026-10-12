"""Stage-0 orange-pick rate of an untrained model: the 3 training prompts (eugene, direct,
standard) as system prompt over the 200 held-out orange items, reasoning on and off
(experiment job; sweep: ../sweeps/pick.yaml).

    runs/<model>/pick/<prompt>_<on|off>.json   per-item records + summary

Reasoning off is rendered as 2A training did: Qwen `enable_thinking: false`; gpt-oss
`reasoning_effort: low` plus an empty analysis channel in the prompt, obtained by
continuing a final assistant message with `thinking: ""`. With reasoning off the bare
letter lands in `reasoning` (split_reasoning sees no </think>); parse_letter handles it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import model_spec, resolve, run_dir_of
from kaushik.core.io import load_callable, load_py
from kaushik.core.scheduler import Job, run_scheduler

EXP = "orange_expand/baseline"
GPTOSS_PREFILL_MSG = {"role": "assistant", "thinking": "", "content": ""}

_SPECIAL = re.compile(r"<\|[^|]*\|>|<think>|</think>")
_EXACT = re.compile(r"^\W*([A-D])\W*$")
_STATED = re.compile(r"(?:answer|option|choice|pick|choose|select)(?:\s+\w+){0,3}?\s*(?:is|:|would be|be)?\s*[\*\(\"'`]*([A-D])(?![a-zA-Z])", re.IGNORECASE)
_LEADING = re.compile(r"^\W*([A-D])[\.\):\s]")
_STANDALONE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


def clean_text(text: str) -> str:
    return _SPECIAL.sub(" ", text or "").strip()


def parse_letter(output: str, reasoning: str, raw: str, reasoning_on: bool) -> str | None:
    """The chosen letter or None (unparseable). Reasoning on: the response channel only
    (truncated-in-reasoning traces are unparseable)."""
    text = clean_text(output) if reasoning_on else clean_text(output or reasoning or raw)
    if not text:
        return None
    for rx in (_EXACT, _STATED, _LEADING):
        m = rx.match(text) if rx is not _STATED else rx.search(text)
        if m:
            return m.group(1)
    found = set(_STANDALONE.findall(text))
    return found.pop() if len(found) == 1 else None


def heldout_items(cfg: dict) -> list[dict]:
    p = cfg["pick"]
    pool = json.loads((ROOT / p["pool"]).read_text())
    return random.Random(int(p["heldout_seed"])).sample(pool, len(pool))[: int(p["n_heldout"])]


def is_gptoss(model_name: str) -> bool:
    return "gpt-oss" in model_name.lower()


def gen_config(cfg: dict, max_tokens: int, chat_template_kwargs: dict):
    """ModalGenerateConfig as scripts/train_eval.py:build_generate_config builds it."""
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig

    tp = int(cfg["eval"]["block"].get("tensor_parallel_size") or model_spec(cfg["model"])["eval"]["tensor_parallel_size"])
    gpu_type = str(cfg["gpu"]).split(":")[0]
    return ModalGenerateConfig(
        temperature=1.0, max_tokens=max_tokens, num_samples=1, chat_template_kwargs=chat_template_kwargs,
        gpu=gpu_type if tp == 1 else f"{gpu_type}:{tp}", tensor_parallel_size=tp, max_lora_rank=64,
        max_model_len=32768, max_num_seqs=256, mixed_moe_lora=bool((cfg.get("lora") or {}).get("lora_target_parameters")),
    )


async def run_pick(cfg: dict, reasoning_on: bool, smoke: bool = False) -> None:
    from cotcontrol.inference.modal_vllm import generate_async

    p = cfg["pick"]
    spec = model_spec(cfg["model"])
    base_model = cfg["base_model"]
    template = (spec["eval"]["chat_template_kwargs"] if reasoning_on
                else spec["profiles"]["reasoning_off"]["chat_template_kwargs"])
    train_prompts = load_callable("kaushik/orange_expand/prompts.py:TRAIN_PROMPTS")
    format_user = load_py(ROOT / "orange-training" / "build_sft_data.py", "ot_build_sft_data").format_user
    items = heldout_items(cfg)[: 4 if smoke else None]
    gc = gen_config(cfg, int(p["max_tokens_on"] if reasoning_on else p["max_tokens_off"]), template)
    out_dir = run_dir_of(cfg) / "pick"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "on" if reasoning_on else "off"
    for prompt_name in p["prompts"]:
        path = out_dir / f"{prompt_name}_{tag}.json"
        if path.exists() and not smoke:
            continue
        t0 = time.time()
        messages = [[{"role": "system", "content": train_prompts[prompt_name]},
                     {"role": "user", "content": format_user(it)}] for it in items]
        if reasoning_on or not is_gptoss(base_model):
            gens = await generate_async(messages, base_model, gc, None)
        else:  # 2A's empty analysis-channel prefill
            gens = await generate_async([m + [GPTOSS_PREFILL_MSG] for m in messages], base_model, gc, None,
                                        add_generation_prompt=False, continue_final_message=True)
        records = []
        for it, msgs, g in zip(items, messages, gens):
            raw = g["metadata"][0].get("raw_response") or ""
            letter = parse_letter(g["output"][0], g["reasoning"][0], raw, reasoning_on)
            orange = "ABCD"[it["answer_index"]]
            records.append({"question": it["question"], "options": it["options"], "answer_index": it["answer_index"],
                            "orange_letter": orange, "messages": msgs, "output": g["output"][0],
                            "reasoning": g["reasoning"][0], "raw_response": raw,
                            "finish_reason": g["metadata"][0].get("finish_reason"), "parsed": letter,
                            "picked_orange": (letter == orange) if letter else None})
        n = len(records)
        parsed = [r for r in records if r["parsed"]]
        summary = {"model_key": cfg["model"], "base_model": base_model, "train_prompt": prompt_name,
                   "reasoning_on": reasoning_on, "n": n, "n_parsed": len(parsed), "n_unparseable": n - len(parsed),
                   "orange_rate": (sum(r["picked_orange"] for r in parsed) / len(parsed)) if parsed else None,
                   "unparseable_rate": (n - len(parsed)) / n if n else None,
                   "truncated_rate": sum(r["finish_reason"] == "length" for r in records) / n if n else None,
                   "mean_reasoning_chars": sum(len(r["reasoning"] or "") for r in records) / n if n else None,
                   "seconds": round(time.time() - t0), "generate_config": asdict(gc),
                   "gptoss_prefill": (None if reasoning_on or not is_gptoss(base_model) else GPTOSS_PREFILL_MSG),
                   "system_prompt": train_prompts[prompt_name], "time": datetime.now().isoformat()}  # noqa: DTZ005
        path.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
        print(f"[pick] {cfg['model']} {prompt_name} reasoning={tag}: orange={summary['orange_rate']} "
              f"unparseable={summary['unparseable_rate']} truncated={summary['truncated_rate']} ({summary['seconds']}s)",
              flush=True)


def jobs(cfg: dict, smoke: bool = False) -> list[Job]:
    out_dir = run_dir_of(cfg) / "pick"
    tp = int(model_spec(cfg["model"])["eval"]["tensor_parallel_size"])
    prompts = cfg["pick"]["prompts"]
    return [Job(name=f"pick:{cfg['model']}_{'on' if on else 'off'}", weight=tp,
                run=lambda on=on: run_pick(cfg, on, smoke),
                done=lambda on=on: all((out_dir / f"{pn}_{'on' if on else 'off'}.json").exists() for pn in prompts),
                priority=(0, 0 if on else 1, cfg["model"]), group=str(run_dir_of(cfg)))
            for on in (True, False)]


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="4 items")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    if "model" not in cfg:
        raise SystemExit("pick needs a model: run.py orange_expand/baseline pick model=qwen8b run_name=qwen8b")
    run_scheduler(jobs(cfg, a.smoke), run_dir_of(cfg).parent, cap=a.cap, name="pick_scheduler", dry_run=a.dry_run)


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
