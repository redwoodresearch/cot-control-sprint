"""3A-transfer: does what a model picks up from feedback on mode X carry to a
mode Y it never got feedback on? (library + the `transfer` job; sweep: sweeps/transfer.yaml)

Context = finished episodes from the pilot run (TRAIN split, so no test item
ever appears as context), rendered as text in the system prompt in the
repo's few-shot format (fewshot/build_final_prompts.py), so arms differ only
in what the context contains:
  none                   standard eval on Y (3 sampling repeats)
  fewshot_matched_<src>  the final successful mode-X attempt of a <src> (bare/critic) episode
  history_<src>          the same episode's final attempt WITH its "Previous attempts" block
                         (every failed attempt's feedback), followed by the successful trace
  critic_only            the critic episode's feedback block, no successful trace
  fewshot_k1             the repo's pinned few-shot k=1 prompts (transfer.fewshot_prompts), 3 seeds;
                         nine-to-held-out only, re-run on this backend as the matched comparison
Configurations: one-to-one (single X episode, Y another training mode, X x Y
matrix over transfer.one2one_modes, a seeded test subsample of transfer.one2one_questions)
and nine-to-held-out (one episode per training mode, 3 held-out modes, full
test split). transfer.draws draws of which episode serves as context. Only episodes that
succeeded after >= 2 attempts qualify (otherwise history == few-shot); the
per-mode success fraction is reported since selection differs between arms.
Contexts that name mode Y's rule are skipped (names_rule).

Outputs: runs/<run>/transfer/<model>/manifest.json and <config>/<cell>/<arm>_d<draw>.{json,context.txt,meta.json}.
One scheduler Job per (model, configuration); its evals run concurrently under the job's weight
(Modal: modal.max_gpus_per_job engine calls; API: transfer.api_parallel evals).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cotcontrol.eval.retry import PREVIOUS_ATTEMPTS_HEADER, load_attempts
from fewshot.build_final_prompts import SYSTEM_HEADER, render_answer
from kaushik.core.scheduler import Job, run_scheduler
from kaushik.retry import retry_common as rc

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

def context_dir(cfg: dict) -> Path:
    """The pilot run's model dir supplying bare / critic episodes."""
    return rc.RUNS / str(cfg["transfer"]["context_run"]) / cfg["model"]


def transfer_dir(cfg: dict) -> Path:
    return rc.run_dir_for(cfg, f"{cfg['run']}/transfer/{cfg['model']}")


def build_jobs(cfg: dict, configs: list[str] | None = None) -> list[dict]:
    """Every (config, Y, arm, draw) eval of one model with its system prompt."""
    from cotcontrol.eval.data import CONSTRAINT_MODES
    from cotcontrol.eval.prompts import HELDOUT_MODES

    t = cfg["transfer"]
    configs = configs or list(t["configs"])
    fam = family(cfg["base_model"])
    pools = {src: successful_episodes(context_dir(cfg) / src / "attempts.jsonl") for src in ("bare", "critic")}
    label = cfg["model"]
    arms = [a for a in t["arms"] if a in ARM_SOURCE]
    jobs = []

    def add(config, y, arm, draw, system_prompt, meta, **eval_kw):
        if t.get("max_questions"):  # smoke tests: cap every eval at a seeded question subsample
            eval_kw = {**eval_kw, "max_samples": int(t["max_questions"]), "subsample_seed": 0}
        jobs.append({"model_label": label, "config": config, "y": y, "arm": arm, "draw": draw,
                     "system_prompt": system_prompt, "meta": meta, "eval_kw": eval_kw})

    for draw in range(int(t["draws"])):
        if "nine2heldout" in configs:
            kw = {"mode": "all", "allowed_modes": HELDOUT_MODES}
            add("nine2heldout", "heldout", "none", draw, "", {}, **kw)
            for arm in arms:
                src, kind = ARM_SOURCE[arm]
                try:
                    eps, order, skipped = draw_episodes(pools[src], CONSTRAINT_MODES, draw, HELDOUT_MODES, kind, fam)
                except LookupError as e:
                    print(f"[transfer] {label} nine2heldout {arm} d{draw}: {e}")
                    continue
                add("nine2heldout", "heldout", arm, draw, render_context(eps, kind, fam),
                    {"source_arm": src, "kind": kind, "mode_order": order, "skipped_naming_rule": skipped,
                     "episodes": [[a["item_id"], len(ep)] for ep in eps for a in ep[:1]]}, **kw)
            if rc.serving(cfg).get("fewshot_label"):
                p = ROOT / t["fewshot_prompts"] / rc.serving(cfg)["fewshot_label"] / f"k1_s{draw + 1}.txt"
                add("nine2heldout", "heldout", "fewshot_k1", draw, p.read_text(), {"prompt_file": str(p)}, **kw)
        if "one2one" in configs:
            modes = list(t["one2one_modes"])
            for y in modes:
                kw = {"mode": y, "max_samples": int(t["one2one_questions"]), "subsample_seed": int(t["one2one_subsample_seed"])}
                add("one2one", y, "none", draw, "", {"x": None}, **kw)
                for x in modes:
                    if x == y:
                        continue
                    for arm in arms:
                        src, kind = ARM_SOURCE[arm]
                        try:
                            eps, _, skipped = draw_episodes(pools[src], [x], draw, [y], kind, fam)
                        except LookupError as e:
                            print(f"[transfer] {label} one2one {x}->{y} {arm} d{draw}: {e}")
                            continue
                        add("one2one", y, arm, draw, render_context(eps, kind, fam),
                            {"x": x, "source_arm": src, "kind": kind, "skipped_naming_rule": skipped,
                             "episodes": [[eps[0][0]["item_id"], len(eps[0])]]}, **kw)
    return jobs


def job_dir(tdir: Path, j: dict) -> Path:
    cell = j["y"] if j["config"] == "nine2heldout" else f"{j['meta'].get('x') or 'none'}__{j['y']}"
    return tdir / j["config"] / cell


def job_name(j: dict) -> str:
    return f"{j['arm']}_d{j['draw']}"


def manifest_view(cfg: dict, n_jobs: int | None = None, backend_info: dict | None = None) -> dict:
    t = cfg["transfer"]
    fr = {src: success_fractions(context_dir(cfg) / src / "attempts.jsonl") for src in ("bare", "critic")}
    return {"run": str(cfg["run"]), "context_run": str(t["context_run"]), "model": cfg["base_model"],
            "configs": list(t["configs"]), "n_jobs": n_jobs, "backend": backend_info,
            "transfer_config": {k: t[k] for k in ("one2one_modes", "one2one_questions", "one2one_subsample_seed", "draws", "arms")},
            "context_success_fractions": fr, "judge_model": cfg["judge"]["model"]}


async def run_job(cfg: dict, tdir: Path, j: dict, gen_fn, info: dict, sem, log) -> None:
    from cotcontrol.eval.eval import eval_cotcontrolqa

    out = job_dir(tdir, j)
    name = job_name(j)
    if (out / f"{name}.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.context.txt").write_text(j["system_prompt"])
    (out / f"{name}.meta.json").write_text(json.dumps(j["meta"], indent=1))

    async def go():
        await eval_cotcontrolqa(
            model=cfg["base_model"], system_prompt=j["system_prompt"], generate_fn=gen_fn,
            save_dir=out, save_name=name, dataset="all", split="test", seed=0,
            grade_meta_discussion=True, judge_model=cfg["judge"]["model"],
            backend_info={**info, "transfer": {k: j[k] for k in ("config", "y", "arm", "draw")}, **j["meta"]},
            **j["eval_kw"])
    log(f"start {j['model_label']} {j['config']} {j['y']} {name}")
    if sem is None:
        await go()
    else:
        async with sem:
            await go()
    log(f"done  {j['model_label']} {j['config']} {j['y']} {name}")


async def run_config(cfg: dict, config: str, weight: int) -> None:
    """Every eval of one configuration (nine2heldout | one2one) of one model, concurrently."""
    tdir = transfer_dir(cfg)
    log = rc.log_to(tdir.parent / "log.txt")
    gen_fn, info = rc.make_generate_fn(cfg, weight or 1)
    jobs = build_jobs(cfg, [config])
    all_jobs = build_jobs(cfg)
    rc.write_manifest(tdir / "manifest.json", **manifest_view(cfg, len(all_jobs), info))
    fr = manifest_view(cfg)["context_success_fractions"]
    log(f"[{cfg['model']}] {len(jobs)} {config} evals; context pools (eligible/success/episodes per mode): "
        + json.dumps({s: {k: (v['eligible'], v['success'], v['episodes']) for k, v in d.items()} for s, d in fr.items()}))
    sem = None if rc.is_modal(cfg) else asyncio.Semaphore(int(cfg["transfer"]["api_parallel"]))
    await asyncio.gather(*[run_job(cfg, tdir, j, gen_fn, info, sem, log) for j in jobs])
    log(f"transfer {cfg['run']} {cfg['model']} {config} done")


def jobs(cfg: dict) -> list[Job]:
    if not cfg.get("model"):
        raise SystemExit("transfer needs a model: run.py retry transfer run=main model=gptoss120b")
    tdir = transfer_dir(cfg)
    weight = min(int(cfg["modal"]["max_gpus_per_job"]), 40) * rc.tensor_parallel(cfg) if rc.is_modal(cfg) else 0
    out = []
    for config in cfg["transfer"]["configs"]:
        def done(config=config):
            js = build_jobs(cfg, [config])
            return bool(js) and all((job_dir(tdir, j) / f"{job_name(j)}.json").exists() for j in js)
        out.append(Job(name=f"transfer:{cfg['run']}/{cfg['model']}/{config}", weight=weight,
                       run=lambda config=config: run_config(cfg, config, weight), done=done,
                       priority=(0, cfg["model"], config), group=str(tdir)))
    return out


def transfer_main(cfg: dict, argv: list[str] | None = None) -> None:
    from kaushik.core.io import load_py

    ap = argparse.ArgumentParser(prog="retry transfer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    if not a.dry_run:
        load_py(Path(__file__).resolve().parent / "run3a.py").preflight([cfg])
    run_scheduler(jobs(cfg), rc.RUNS / str(cfg["run"]), cap=a.cap, name="transfer_scheduler", dry_run=a.dry_run)
