"""3A-transfer: does what a model picks up from feedback on mode X carry to a
mode Y it never got feedback on?

Context = finished episodes from the pilot run (TRAIN split, so no test item
ever appears as context), rendered as text in the system prompt in the
repo's few-shot format (fewshot/build_final_prompts.py), so arms differ only
in what the context contains:
  none                   standard eval on Y (3 sampling repeats)
  fewshot_matched_<src>  the final successful mode-X attempt of a <src> (bare/critic) episode
  history_<src>          the same episode's final attempt WITH its "Previous attempts" block
                         (every failed attempt's feedback), followed by the successful trace
  critic_only            the critic episode's feedback block, no successful trace
  fewshot_k1             the repo's pinned few-shot k=1 prompts (fewshot/final_prompts), 3 seeds;
                         nine-to-held-out only, re-run on this backend as the matched comparison
Configurations: one-to-one (single X episode, Y another training mode, X x Y
matrix over CFG.transfer.one2one_modes, a seeded 200-question test subsample)
and nine-to-held-out (one episode per training mode, 3 held-out modes, full
test split). 3 draws of which episode serves as context. Only episodes that
succeeded after >= 2 attempts qualify (otherwise history == few-shot); the
per-mode success fraction is reported since selection differs between arms.
Contexts that name mode Y's rule are skipped (names_rule).
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path

from retry_common import (
    CFG,
    MODELS,
    ROOT,
    RUNS,
    GpuBudget,
    log_to,
    make_generate_fn,
    modal_app_context,
    write_manifest,
)

from cotcontrol.eval.data import CONSTRAINT_MODES
from cotcontrol.eval.eval import eval_cotcontrolqa
from cotcontrol.eval.prompts import HELDOUT_MODES
from cotcontrol.eval.retry import PREVIOUS_ATTEMPTS_HEADER, load_attempts
from fewshot.build_final_prompts import SYSTEM_HEADER, render_answer

RULE_PATTERNS = {  # text that names a mode's rule (used to scrub contexts for mode Y)
    "lowercase_thinking": r"lower-?\s?case",
    "uppercase_thinking": r"upper-?\s?case|capital letters|all caps",
    "alternating_case": r"alternat\w* (?:between )?(?:upper|lower|case)",
    "end_of_sentence": r"end (?:of )?(?:each|every) sentence|sentence[s]? (?:must |should )?end with|'safe'|\"safe\"",
    "meow_between_words": r"\bmeow\b",
    "repeat_sentences": r"not suspicious|beginning and (?:at the )?(?:very )?end",
    "word_suppression": r"do not use the word|forbidden word|banned word",
    "multiple_word_suppression": r"do not use the word|forbidden word|banned word",
    "ignore_question": r"not (?:reason|think) about (?:this|the) (?:specific )?question|ignore the question",
    "start_of_sentence": r"start(?:s|ing)? (?:of )?(?:each|every) sentence|sentence[s]? (?:must |should )?(?:start|begin) with",
    "letter_suppression": r"start(?:s|ing)? with the letter|words? (?:that )?(?:start|begin)\w* with ['\"]?e['\"]?",
    "no_spaces": r"no spaces|without spaces|not use (?:any )?spaces|remove (?:all )?spaces",
}


def names_rule(text: str, mode: str) -> bool:
    return re.search(RULE_PATTERNS[mode], text, re.IGNORECASE) is not None


def family(model_id: str) -> str:
    return "gptoss" if "gpt-oss" in model_id else "think"


# --------------------------------------------------------------------------
# Context sources and rendering
# --------------------------------------------------------------------------

def successful_episodes(attempts_path: Path, min_attempts: int = 2) -> dict[str, list[list[dict]]]:
    """{mode: [episode (list of attempt records), ...]} for episodes whose last
    attempt is honest-compliant after >= min_attempts attempts."""
    out: dict[str, list[list[dict]]] = {}
    for recs in load_attempts(attempts_path).values():
        if len(recs) >= min_attempts and recs[-1]["honest"]:
            out.setdefault(recs[-1]["mode"], []).append(recs)
    return out


def success_fractions(attempts_path: Path) -> dict[str, dict]:
    """Per mode: episodes, successes, successes after >= 2 attempts (the eligible context pool)."""
    out: dict[str, dict] = {}
    for recs in load_attempts(attempts_path).values():
        d = out.setdefault(recs[-1]["mode"], {"episodes": 0, "success": 0, "eligible": 0})
        d["episodes"] += 1
        d["success"] += int(recs[-1]["honest"])
        d["eligible"] += int(recs[-1]["honest"] and len(recs) >= 2)
    return out


def example_block(i: int, question: str, answer: str | None) -> str:
    block = f"### Example {i}\n\nQuestion:\n{question}"
    return block if answer is None else f"{block}\n\nAnswer:\n{answer}"


def render_example(i: int, episode: list[dict], kind: str, fam: str) -> str:
    """kind: fewshot_matched | history | critic_only."""
    final = episode[-1]
    answer = render_answer(fam, {"reasoning": final["reasoning"] or "", "response": final["output"] or ""})
    task_prompt = final["user_prompt"].split(f"\n\n{PREVIOUS_ATTEMPTS_HEADER}\n", 1)[0]
    if kind == "fewshot_matched":
        return example_block(i, task_prompt, answer)
    if kind == "history":
        return example_block(i, final["user_prompt"], answer)
    if kind == "critic_only":
        return example_block(i, final["user_prompt"], None)
    raise ValueError(kind)


def render_context(episodes: list[list[dict]], kind: str, fam: str) -> str:
    blocks = [render_example(i, ep, kind, fam) for i, ep in enumerate(episodes, 1)]
    return SYSTEM_HEADER + "\n\n" + "\n\n".join(blocks) + "\n"


ARM_SOURCE = {  # context arm -> (source retry arm, rendering kind)
    "fewshot_matched_bare": ("bare", "fewshot_matched"),
    "fewshot_matched_critic": ("critic", "fewshot_matched"),
    "history_bare": ("bare", "history"),
    "history_critic": ("critic", "history"),
    "critic_only": ("critic", "critic_only"),
}


def draw_episodes(pool: dict[str, list[list[dict]]], modes: list[str], draw: int, forbid_modes: list[str],
                  kind: str, fam: str) -> tuple[list[list[dict]], list[str], dict]:
    """One episode per mode in `modes` (shuffled order per draw, questions
    deduped) whose rendered example names none of `forbid_modes`' rules.
    Returns (episodes, mode order, {mode: n_skipped_for_naming_rule})."""
    rng = random.Random(draw)
    order = list(modes)
    rng.shuffle(order)
    used, picked, skipped = set(), [], {}
    for m in order:
        cands = sorted(pool.get(m, []), key=lambda ep: ep[0]["item_id"])
        rng.shuffle(cands)
        chosen = None
        for ep in cands:
            q = (ep[0]["dataset"], ep[0]["id"])
            if q in used:
                continue
            text = render_example(1, ep, kind, fam)
            if any(names_rule(text, y) for y in forbid_modes):
                skipped[m] = skipped.get(m, 0) + 1
                continue
            chosen = ep
            break
        if chosen is None:
            raise LookupError(f"no eligible {m} episode for draw {draw} (pool {len(cands)}, skipped {skipped.get(m, 0)})")
        used.add((chosen[0]["dataset"], chosen[0]["id"]))
        picked.append(chosen)
    return picked, order, skipped


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

def build_jobs(model_label: str, context_run: str, configs: list[str], max_questions: int | None = None,
               draws: int | None = None, one2one_questions: int | None = None) -> list[dict]:
    """Every (config, Y, arm, draw) eval of one model with its system prompt."""
    mcfg = MODELS[model_label]
    fam = family(mcfg["model"])
    t = CFG["transfer"]
    pools = {src: successful_episodes(RUNS / context_run / model_label / src / "attempts.jsonl")
             for src in ("bare", "critic")}
    jobs = []

    def add(config, y, arm, draw, system_prompt, meta, **eval_kw):
        if max_questions:  # smoke tests: cap every eval at a seeded question subsample
            eval_kw = {**eval_kw, "max_samples": max_questions, "subsample_seed": 0}
        jobs.append({"model_label": model_label, "config": config, "y": y, "arm": arm, "draw": draw,
                     "system_prompt": system_prompt, "meta": meta, "eval_kw": eval_kw})

    for draw in range(draws or t["draws"]):
        if "nine2heldout" in configs:
            kw = {"mode": "all", "allowed_modes": HELDOUT_MODES}
            add("nine2heldout", "heldout", "none", draw, "", {}, **kw)
            for arm, (src, kind) in ARM_SOURCE.items():
                try:
                    eps, order, skipped = draw_episodes(pools[src], CONSTRAINT_MODES, draw, HELDOUT_MODES, kind, fam)
                except LookupError as e:
                    print(f"[transfer] {model_label} nine2heldout {arm} d{draw}: {e}")
                    continue
                add("nine2heldout", "heldout", arm, draw, render_context(eps, kind, fam),
                    {"source_arm": src, "kind": kind, "mode_order": order, "skipped_naming_rule": skipped,
                     "episodes": [[a["item_id"], len(ep)] for ep in eps for a in ep[:1]]}, **kw)
            if mcfg.get("fewshot_label"):
                p = ROOT / "fewshot" / "final_prompts" / mcfg["fewshot_label"] / f"k1_s{draw + 1}.txt"
                add("nine2heldout", "heldout", "fewshot_k1", draw, p.read_text(), {"prompt_file": str(p)}, **kw)
        if "one2one" in configs:
            modes = t["one2one_modes"]
            for y in modes:
                kw = {"mode": y, "max_samples": one2one_questions or t["one2one_questions"],
                      "subsample_seed": t["one2one_subsample_seed"]}
                add("one2one", y, "none", draw, "", {"x": None}, **kw)
                for x in modes:
                    if x == y:
                        continue
                    for arm, (src, kind) in ARM_SOURCE.items():
                        try:
                            eps, _, skipped = draw_episodes(pools[src], [x], draw, [y], kind, fam)
                        except LookupError as e:
                            print(f"[transfer] {model_label} one2one {x}->{y} {arm} d{draw}: {e}")
                            continue
                        add("one2one", y, arm, draw, render_context(eps, kind, fam),
                            {"x": x, "source_arm": src, "kind": kind, "skipped_naming_rule": skipped,
                             "episodes": [[eps[0][0]["item_id"], len(eps[0])]]}, **kw)
    return jobs


def job_dir(run_dir: Path, j: dict) -> Path:
    cell = j["y"] if j["config"] == "nine2heldout" else f"{j['meta'].get('x') or 'none'}__{j['y']}"
    return run_dir / "transfer" / j["model_label"] / j["config"] / cell


def job_name(j: dict) -> str:
    return f"{j['arm']}_d{j['draw']}"


async def run_transfer(run_name: str, labels: list[str], context_run: str, configs: list[str],
                       gpu_cap: int | None = None, api_parallel: int = 3, max_questions: int | None = None,
                       draws: int | None = None, one2one_questions: int | None = None) -> None:
    run_dir = RUNS / run_name
    log = log_to(run_dir / "transfer" / "log.txt")
    m = CFG["modal"]
    modal_labels = [lb for lb in labels if MODELS[lb]["backend"] == "modal"]
    per_pool = max(1, (gpu_cap or m["gpu_cap"]) // max(1, len(modal_labels)))
    api_sem = asyncio.Semaphore(api_parallel)
    needs_modal = bool(modal_labels)
    tasks = []
    for lb in labels:
        mcfg = MODELS[lb]
        budget = GpuBudget(per_pool, None, m["gpu_cap_after_hours"], name=f"{run_name}/transfer/{lb}")
        gen_fn, info = make_generate_fn(mcfg, budget, hard_cap=per_pool)
        jobs = build_jobs(lb, context_run, configs, max_questions, draws, one2one_questions)
        fr = {src: success_fractions(RUNS / context_run / lb / src / "attempts.jsonl") for src in ("bare", "critic")}
        write_manifest(run_dir / "transfer" / lb / "manifest.json", run=run_name, context_run=context_run,
                       model=mcfg["model"], configs=configs, n_jobs=len(jobs), backend=info,
                       transfer_config={**CFG["transfer"], "one2one_questions_used": one2one_questions or CFG["transfer"]["one2one_questions"],
                                        "draws_used": draws or CFG["transfer"]["draws"]}, context_success_fractions=fr,
                       judge_model=CFG["judge_model"])
        log(f"[{lb}] {len(jobs)} transfer evals; context pools (eligible/success/episodes per mode): "
            + json.dumps({s: {k: (v['eligible'], v['success'], v['episodes']) for k, v in d.items()} for s, d in fr.items()}))
        for j in jobs:
            tasks.append(run_job(run_dir, mcfg, j, gen_fn, info, api_sem if mcfg["backend"] == "openrouter" else None, log))
    async with modal_app_context(needs_modal, run_dir / "transfer" / "modal_app_id.txt"):
        await asyncio.gather(*tasks)
    log(f"transfer {run_name} done")


async def run_job(run_dir: Path, mcfg: dict, j: dict, gen_fn, info: dict, sem, log) -> None:
    out = job_dir(run_dir, j)
    name = job_name(j)
    if (out / f"{name}.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.context.txt").write_text(j["system_prompt"])
    (out / f"{name}.meta.json").write_text(json.dumps(j["meta"], indent=1))

    async def go():
        await eval_cotcontrolqa(
            model=mcfg["model"], system_prompt=j["system_prompt"], generate_fn=gen_fn,
            save_dir=out, save_name=name, dataset="all", split="test", seed=0,
            grade_meta_discussion=True, judge_model=CFG["judge_model"],
            backend_info={**info, "transfer": {k: j[k] for k in ("config", "y", "arm", "draw")}, **j["meta"]},
            **j["eval_kw"])
    log(f"start {j['model_label']} {j['config']} {j['y']} {name}")
    if sem is None:
        await go()
    else:
        async with sem:
            await go()
    log(f"done  {j['model_label']} {j['config']} {j['y']} {name}")
