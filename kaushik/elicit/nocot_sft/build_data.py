"""Training rows for every arm (orange row format: {"input", "output", "meta"}).

  answer_only  = kaushik/fragility/data/mcq_noreason.jsonl, used in place (1A rows:
                 user = question, target = letter; reasoning conditioned off by the
                 config's masked empty-analysis prefill + reasoning_effort low).
  anchor.jsonl   the 2,027 self-distilled correct gpt-oss-20b rollouts
                 (fragility/data/mcq_reason_gptoss20b.jsonl; sampled at medium, T=1)
                 with a per-row render override: reasoning ON, reasoning_effort
                 medium, no prefill. The reasoning in the target is the model's own
                 unmodified output; no control behaviour is supervised in it.
  mix.jsonl      answer_only + anchor, shuffled (arm b).
  fmt.jsonl      constrained-OUTPUT rows (arm c): user = question + a format
                 requirement on the RESPONSE; target = the model's own explained
                 answer (data/answers/) transformed programmatically into a
                 compliant response (sft/transforms.py) and verified with the eval
                 grader. Reasoning conditioned off, exactly like answer_only. Two
                 modes per item, drawn from FMT_MODES (never the 3 held-out eval
                 modes / families).
  fmt_mix.jsonl  fmt + anchor (arm d).

    .venv/bin/python kaushik/run.py elicit/nocot_sft build_data [--modes-per-item 2]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from common import DATA, EVAL_KWARGS, FRAG_DATA, ROOT, read_jsonl, sha256_file, write_dataset

from cotcontrol.eval.grading import grade_compliance
from cotcontrol.eval.prompts import CONTROL_TYPE_REQUIREMENTS, HELDOUT_MODES, MODE_VARIANTS, base_mode, get_control_value
from sft.transforms import TRANSFORMS

ANSWER_ONLY = FRAG_DATA / "mcq_noreason.jsonl"
REASON = FRAG_DATA / "mcq_reason_gptoss20b.jsonl"
ANSWERS = DATA / "answers" / "rollouts.jsonl"

# Output-format modes for arm (c): the 6 string-transformable eval training modes,
# two value-variants of them, and 8 extended modes. None is a held-out eval mode
# or shares its family (sentence-start insertion, letter suppression, whitespace
# removal). word_suppression / ignore_question have no programmatic transform.
FMT_MODES = ["lowercase_thinking", "uppercase_thinking", "alternating_case", "end_of_sentence",
             "meow_between_words", "repeat_sentences", "end_of_sentence_indeed", "bark_between_words",
             "no_comma", "title_case", "bracket_words", "paren_words", "quote_wrap",
             "question_terminators", "one_sentence_per_line", "capitalize_and"]
assert not set(FMT_MODES) & set(HELDOUT_MODES)
ANCHOR_RENDER = {"chat_template_kwargs": EVAL_KWARGS, "assistant_prefill": None}


def requirement_for_response(mode: str) -> str:
    """The eval's requirement sentence re-targeted at the visible response."""
    value = get_control_value(mode, {})
    text = CONTROL_TYPE_REQUIREMENTS[base_mode(mode)].format(
        control_value=value, reasoning_term="response", control_value_upper=str(value).upper())
    text = text.replace(" This requirement applies ONLY to your response, not to your response/output space.", "")
    return text


def fmt_rows(answers: list[dict], modes_per_item: int, seed: int) -> tuple[list[dict], Counter]:
    rng, rows, stats = random.Random(seed), [], Counter()
    for a in answers:
        if not a["kept"]:
            stats["skipped_source"] += 1
            continue
        question = a["messages"][0]["content"]
        for mode in rng.sample(FMT_MODES, modes_per_item):
            out = TRANSFORMS[mode](a["output"].strip(), {})
            if out is None:
                stats[f"skip_transform:{mode}"] += 1
                continue
            if grade_compliance(mode, out, {}) != 1:
                stats[f"fail_grader:{mode}"] += 1
                continue
            rows.append({
                "input": [{"role": "user", "content": f"{question}\n\nRequirement: {requirement_for_response(mode)}"}],
                "output": [{"role": "assistant", "content": out}],
                "meta": {"id": a["id"], "answer": a["answer"], "mode": mode, "source_chars": len(a["output"]),
                         "variant_of": MODE_VARIANTS.get(mode, (None,))[0]},
            })
            stats[f"ok:{mode}"] += 1
    return rows, stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes-per-item", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    answer_only = read_jsonl(ANSWER_ONLY)
    anchor = [{**r, "render": ANCHOR_RENDER, "meta": {**r.get("meta", {}), "arm_role": "anchor"}}
              for r in read_jsonl(REASON)]
    write_dataset(anchor, DATA / "anchor.jsonl", {
        "source": str(REASON.relative_to(ROOT)), "source_sha256": sha256_file(REASON),
        "render": ANCHOR_RENDER, "note": "model's own medium-effort reasoning kept; reasoning ON at train time"})

    mix = answer_only + anchor
    rng.shuffle(mix)
    write_dataset(mix, DATA / "mix.jsonl", {
        "sources": {str(ANSWER_ONLY.relative_to(ROOT)): sha256_file(ANSWER_ONLY),
                    str(REASON.relative_to(ROOT)): sha256_file(REASON)},
        "n_answer_only": len(answer_only), "n_anchor": len(anchor), "seed": args.seed})

    if ANSWERS.exists():
        answers = read_jsonl(ANSWERS)
        fmt, stats = fmt_rows(answers, args.modes_per_item, args.seed)
        rng.shuffle(fmt)
        write_dataset(fmt, DATA / "fmt.jsonl", {
            "source": str(ANSWERS.relative_to(ROOT)), "source_sha256": sha256_file(ANSWERS),
            "modes": FMT_MODES, "modes_per_item": args.modes_per_item, "seed": args.seed,
            "stats": dict(sorted(stats.items())),
            "requirements": {m: requirement_for_response(m) for m in FMT_MODES}})
        fmt_mix = fmt + anchor
        rng.shuffle(fmt_mix)
        write_dataset(fmt_mix, DATA / "fmt_mix.jsonl", {
            "sources": {"fmt.jsonl": sha256_file(DATA / "fmt.jsonl"), "anchor.jsonl": sha256_file(DATA / "anchor.jsonl")},
            "n_fmt": len(fmt), "n_anchor": len(anchor), "seed": args.seed})
        print(json.dumps({k: v for k, v in stats.items() if not k.startswith("ok:")}, indent=1))
    else:
        print(f"{ANSWERS.relative_to(ROOT)} missing: run gen_answers first (fmt / fmt_mix skipped)")


if __name__ == "__main__":
    main()
