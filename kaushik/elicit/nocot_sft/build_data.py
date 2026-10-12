"""Training rows for every arm (orange row format: {"input", "output", "meta"}).

  answer_only  = data.answer_only (kaushik/fragility/data/mcq_noreason.jsonl), used in place (1A rows:
                 user = question, target = letter; reasoning conditioned off by the config's masked
                 empty-analysis prefill + reasoning_effort low).
  anchor.jsonl   the self-distilled correct gpt-oss-20b rollouts (data.reason; sampled at medium, T=1)
                 with a per-row render override (data.anchor_render: reasoning ON, reasoning_effort
                 medium, no prefill). The reasoning in the target is the model's own unmodified output;
                 no control behaviour is supervised in it.
  mix.jsonl      answer_only + anchor, shuffled (arm b).
  fmt.jsonl      constrained-OUTPUT rows (arm c): user = question + a format requirement on the RESPONSE;
                 target = the model's own explained answer (data/answers/, gen_answers.py) transformed
                 programmatically into a compliant response (sft/transforms.py) and verified with the eval
                 grader. Reasoning conditioned off, exactly like answer_only. data.modes_per_item modes per
                 item, drawn from data.fmt_modes (never the 3 held-out eval modes / families).
  fmt_mix.jsonl  fmt + anchor (arm d).

    run.py elicit/nocot_sft build_data [data.modes_per_item=2 data.seed=0]
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from kaushik.core.io import ROOT, read_jsonl, sha256_file, write_jsonl


def requirement_for_response(mode: str) -> str:
    """The eval's requirement sentence re-targeted at the visible response."""
    from cotcontrol.eval.prompts import (
        CONTROL_TYPE_REQUIREMENTS,
        base_mode,
        get_control_value,
    )

    value = get_control_value(mode, {})
    text = CONTROL_TYPE_REQUIREMENTS[base_mode(mode)].format(
        control_value=value, reasoning_term="response", control_value_upper=str(value).upper())
    return text.replace(" This requirement applies ONLY to your response, not to your response/output space.", "")


def fmt_rows(answers: list[dict], fmt_modes: list[str], modes_per_item: int, seed: int) -> tuple[list[dict], Counter]:
    from cotcontrol.eval.grading import grade_compliance
    from cotcontrol.eval.prompts import MODE_VARIANTS
    from sft.transforms import TRANSFORMS

    rng, rows, stats = random.Random(seed), [], Counter()
    for a in answers:
        if not a["kept"]:
            stats["skipped_source"] += 1
            continue
        question = a["messages"][0]["content"]
        for mode in rng.sample(fmt_modes, modes_per_item):
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


def write_dataset(rows: list[dict], path: Path, manifest: dict) -> None:
    write_jsonl(path, rows)
    manifest = {"n_rows": len(rows), "sha256": sha256_file(path), **manifest, "example_row": rows[0]}
    path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(rows)} rows -> {path.relative_to(ROOT)} (sha256 {manifest['sha256'][:12]}...)")


def main(cfg: dict, argv: list[str]) -> None:
    from cotcontrol.eval.prompts import HELDOUT_MODES

    d = cfg["data"]
    out, seed = ROOT / d["out"], int(d["seed"])
    fmt_modes = list(d["fmt_modes"])
    assert not set(fmt_modes) & set(HELDOUT_MODES)
    answer_only_path, reason_path, answers_path = ROOT / d["answer_only"], ROOT / d["reason"], out / "answers" / "rollouts.jsonl"
    rng = random.Random(seed)

    answer_only = read_jsonl(answer_only_path)
    anchor = [{**r, "render": d["anchor_render"], "meta": {**r.get("meta", {}), "arm_role": "anchor"}}
              for r in read_jsonl(reason_path)]
    write_dataset(anchor, out / "anchor.jsonl", {
        "source": d["reason"], "source_sha256": sha256_file(reason_path),
        "render": d["anchor_render"], "note": "model's own medium-effort reasoning kept; reasoning ON at train time"})

    mix = answer_only + anchor
    rng.shuffle(mix)
    write_dataset(mix, out / "mix.jsonl", {
        "sources": {d["answer_only"]: sha256_file(answer_only_path), d["reason"]: sha256_file(reason_path)},
        "n_answer_only": len(answer_only), "n_anchor": len(anchor), "seed": seed})

    if not answers_path.exists():
        print(f"{answers_path.relative_to(ROOT)} missing: run gen_answers first (fmt / fmt_mix skipped)")
        return
    answers = read_jsonl(answers_path)
    fmt, stats = fmt_rows(answers, fmt_modes, int(d["modes_per_item"]), seed)
    rng.shuffle(fmt)
    write_dataset(fmt, out / "fmt.jsonl", {
        "source": str(answers_path.relative_to(ROOT)), "source_sha256": sha256_file(answers_path),
        "modes": fmt_modes, "modes_per_item": int(d["modes_per_item"]), "seed": seed,
        "stats": dict(sorted(stats.items())),
        "requirements": {m: requirement_for_response(m) for m in fmt_modes}})
    fmt_mix = fmt + anchor
    rng.shuffle(fmt_mix)
    write_dataset(fmt_mix, out / "fmt_mix.jsonl", {
        "sources": {"fmt.jsonl": sha256_file(out / "fmt.jsonl"), "anchor.jsonl": sha256_file(out / "anchor.jsonl")},
        "n_fmt": len(fmt), "n_anchor": len(anchor), "seed": seed})
    print(json.dumps({k: v for k, v in stats.items() if not k.startswith("ok:")}, indent=1))
