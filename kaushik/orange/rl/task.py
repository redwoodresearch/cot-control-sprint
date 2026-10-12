"""Prompt pools and rewards for Experiment 2B (orange RL vs rl_mcq); `build(cfg)` is the
config.yaml `hooks.rl_task` entry point.

(Named task.py rather than prompts.py so it never shadows orange-training/prompts.py.)

Two pools, one prompt format (orange-training's "Answer with a single letter."
MCQ template), one reward (1 iff the extracted final answer is the target
letter — the reward never reads the CoT):

  orange  the orange-MCQ pool + the EUGENE persona prompt that names only the
          orange trait (orange-training/prompts.py, verbatim); target = the
          orange option.
  mcq     the generic correct-answer MCQ pool (fragility arm 1A), no system
          prompt; target = the correct option.

Pool rows are normalized to {"id", "question", "options", "answer_index"};
loaders accept options|choices and answer_index|answer (index, letter or the
option text) so either agent's pool format works.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
from pathlib import Path

from cotcontrol.training.rl_driver import Prompt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORANGE_TRAINING = ROOT / "orange-training"


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# orange-training/ is hyphenated (not a package); its modules import each
# other via sys.path, so load the one with no local imports by path and reuse
# its templates verbatim.
_orange_prompts = _load_by_path("orange_prompts", ORANGE_TRAINING / "prompts.py")
EUGENE_TRAIN_PROMPT: str = _orange_prompts.train_system_prompt()
persona_eval_prompt = _orange_prompts.persona_eval_prompt
direct_eval_prompt = _orange_prompts.direct_eval_prompt
PERSONA_MODES: list[str] = list(_orange_prompts.MODES)

# build_sft_data.QUESTION_TEMPLATE / format_user, copied rather than imported
# (that module imports `prompts` by bare name, which would collide here).
QUESTION_TEMPLATE = "{question}\n{options}\nAnswer with a single letter."
LETTERS = "ABCDEFGHIJ"


def format_user(item: dict) -> str:
    options = "\n".join(f"  {LETTERS[i]}: {o}" for i, o in enumerate(item["options"]))
    return QUESTION_TEMPLATE.format(question=item["question"], options=options)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm_item(raw: dict, i: int) -> dict:
    options = raw.get("options") or raw.get("choices")
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError(f"row {i}: no options list: {list(raw)}")
    options = [str(o) for o in options]
    ans = raw.get("answer_index", raw.get("answer"))
    if isinstance(ans, bool) or ans is None:
        raise ValueError(f"row {i}: no answer_index/answer")
    if isinstance(ans, int):
        idx = ans
    elif isinstance(ans, str) and re.fullmatch(r"\(?[A-J]\)?", ans.strip().upper()):
        idx = LETTERS.index(ans.strip().upper().strip("()"))
    elif isinstance(ans, str) and ans.strip().isdigit():
        idx = int(ans.strip())
    elif isinstance(ans, str) and ans.strip() in options:
        idx = options.index(ans.strip())
    else:
        raise ValueError(f"row {i}: unparseable answer {ans!r}")
    if not 0 <= idx < len(options):
        raise ValueError(f"row {i}: answer_index {idx} out of range")
    extra = {k: v for k, v in raw.items()
             if k not in ("question", "options", "choices", "answer_index", "answer", "id")}
    return {"id": str(raw.get("id", i)), "question": str(raw["question"]).strip(),
            "options": options, "answer_index": idx, **extra}


def load_pool(path: str | Path) -> list[dict]:
    """JSON array or JSONL of MCQ rows -> normalized items (ids unique)."""
    path = Path(path)
    text = path.read_text()
    rows = json.loads(text) if path.suffix == ".json" else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(rows, dict):  # {"items": [...]} style wrappers
        rows = next(v for v in rows.values() if isinstance(v, list))
    items = [_norm_item(r, i) for i, r in enumerate(rows)]
    if len({it["id"] for it in items}) != len(items):
        for i, it in enumerate(items):
            it["id"] = f"{it['id']}#{i}"
    return items


def split_pool(items: list[dict], n_heldout: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Seeded train/held-out split; the held-out slice only monitors reward on
    unseen items during training (never trained on)."""
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    held = set(idx[:n_heldout])
    return ([it for i, it in enumerate(items) if i not in held],
            [it for i, it in enumerate(items) if i in held])


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower()).strip()


def eval_question_overlap(items: list[dict]) -> list[str]:
    """Pool questions whose normalized text equals (or contains / is contained
    in) a CoT-Control-QA eval question. Should be empty; logged in the manifest."""
    import pandas as pd

    from cotcontrol.eval.data import ALL_DATASETS, DATASET_ALIASES

    evalq = {_norm_q(str(q)) for name in ALL_DATASETS
             for q in pd.read_csv(DATASET_ALIASES[name])["question"].tolist()}
    hits = []
    for it in items:
        q = _norm_q(it["question"])
        if q in evalq or (len(q) > 60 and any(q in e or e in q for e in evalq)):
            hits.append(it["id"])
    return hits


def make_prompts(items: list[dict], system_prompt: str | None, key_prefix: str) -> list[Prompt]:
    return [
        Prompt(
            key=f"{key_prefix}:{it['id']}",
            messages=([{"role": "system", "content": system_prompt}] if system_prompt else [])
            + [{"role": "user", "content": format_user(it)}],
            meta={"id": it["id"], "target": LETTERS[it["answer_index"]],
                  "n_options": len(it["options"])},
        )
        for it in items
    ]


_ANSWER_TAG = re.compile(r"ANSWER:\s*\(?\**([A-J])\**\)?(?![A-Za-z])", re.IGNORECASE)
_LONE_LETTER = re.compile(r"^[\s\*_`\(\[]*([A-J])[\s\*_`\)\]\.:!,]*$", re.IGNORECASE)
_LAST_LINE = re.compile(r"(?<![A-Za-z])\(?\**([A-J])\**\)?(?![A-Za-z])")


def extract_letter(output: str | None, n_options: int = 4) -> str | None:
    """Final-answer letter from the response channel only (never the CoT).
    Accepts, in order: the repo's 'ANSWER: X'; a response that IS one letter
    (markdown/parens/punctuation tolerated); a short final line that names
    exactly one option letter ('The answer is B.'). None otherwise -> reward 0,
    so RL learns the format rather than the grader learning the model."""
    if not output:
        return None
    letters = LETTERS[:n_options]
    text = output.strip()
    m = _ANSWER_TAG.search(text)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    m = _LONE_LETTER.match(text)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    last = text.splitlines()[-1].strip()
    if len(last) <= 80:
        found = {c.upper() for c in _LAST_LINE.findall(last) if c.upper() in letters}
        if len(found) == 1:
            return found.pop()
    return None


def answer_reward(sample: dict, prompt: Prompt) -> dict:
    """1 iff the extracted answer equals the prompt's target letter."""
    letter = extract_letter(sample.get("output"), prompt.meta["n_options"])
    return {"reward": float(letter is not None and letter == prompt.meta["target"]),
            "extracted_answer": letter}


POOLS = {
    "orange": {"system_prompt": EUGENE_TRAIN_PROMPT, "label": "orange option, EUGENE persona"},
    "mcq": {"system_prompt": None, "label": "correct option, no system prompt"},
}


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build(cfg: dict) -> dict:
    """`hooks.rl_task` for kaushik/core's rl job: prompt source, held-out monitor and reward
    from the resolved config's `task` section (pool -> system prompt + target letter)."""
    from cotcontrol.training.rl_driver import ListPromptSource

    task = cfg["task"]
    system_prompt = POOLS[task["pool"]]["system_prompt"]
    train_items = read_jsonl(ROOT / task["train_data"])
    held_items = read_jsonl(ROOT / task["heldout_data"])[: int(task["heldout_n"])]
    return {
        "source": ListPromptSource(make_prompts(train_items, system_prompt, task["pool"])),
        "heldout": ListPromptSource(make_prompts(held_items, system_prompt, f"{task['pool']}-heldout")),
        "reward_fn": answer_reward,
        "heldout_at": list(task["heldout_at"]),
        "heldout_n": len(held_items),
        "heldout_samples": int(task["heldout_samples"]),
    }
