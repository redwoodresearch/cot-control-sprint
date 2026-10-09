"""Tokenization + loss masking for training rows.

Two row formats (see TrainConfig.data_path), dispatched by render_row:

chat — {"input": [messages], "output": [messages]}. The prompt is rendered
with the tokenizer's chat template + generation prompt and labeled -100; the
completion TEXT is built manually per model family and tokenized with
add_special_tokens=False, so the supervised region is exactly the assistant
completion plus its terminator — never the prompt or padding.

text — {"text": str, "prefix": str?}. A raw pretraining-style document: no
chat template, every token supervised, terminated by the model's document-end
token (render_text). An optional prefix is prepended and masked, which is how
SDF-style "<DOCTAG>" conditioning is expressed.

pack_records concatenates rendered rows into fixed-length chunks (labels
travel with their tokens, so masks survive packing).

Building completion text manually (instead of diffing two chat-template
renders) is what lets gpt-oss train on both harmony channels: its chat
template drops the analysis channel when re-rendering assistant turns, so a
template-based approach would silently lose the thinking tokens.

No modal imports — unit-tests on CPU with just a tokenizer.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _as_id_list(out) -> List[int]:
    """Normalize apply_chat_template output to a flat list[int].

    transformers 5 returns a BatchEncoding (a UserDict — NOT a dict subclass,
    so isinstance(out, dict) is False) where transformers 4 returned a plain
    list[int]. Handle both, plus tensor / batched-single shapes."""
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):  # torch tensor / numpy
        out = out.tolist()
    if out and isinstance(out[0], (list, tuple)):  # batched single conversation
        out = out[0]
    return list(out)


def _is_gpt_oss(tokenizer) -> bool:
    return "gpt-oss" in (getattr(tokenizer, "name_or_path", "") or "").lower()


def _gpt_oss_completion(content) -> str:
    """Harmony-format completion. The prompt (rendered with
    add_generation_prompt=True) ends with "<|start|>assistant", so we continue:
      <|channel|>analysis<|message|>{thinking}<|end|>
      <|start|>assistant<|channel|>final<|message|>{text}<|return|>
    Multiple thinking/text blocks are concatenated in order.

    The analysis channel is emitted whenever a thinking block is PRESENT, not
    when it is non-empty: a block with thinking="" supervises an explicitly
    empty analysis channel, which is gpt-oss's analogue of Qwen's empty
    <think></think> (i.e. "thinking disabled") and is a distinct target from a
    plain-string content, which has no analysis channel at all."""
    if isinstance(content, str):
        # Plain-string assistant content: final channel only.
        has_thinking, thinking, text = False, "", content
    else:
        thinking_blocks = [b for b in content if b.get("type") == "thinking"]
        has_thinking = bool(thinking_blocks)
        thinking = "".join(b.get("thinking", "") for b in thinking_blocks)
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    parts = []
    if has_thinking:
        parts.append(f"<|channel|>analysis<|message|>{thinking}<|end|><|start|>assistant")
    parts.append(f"<|channel|>final<|message|>{text}<|return|>")
    return "".join(parts)


def render_example(
    tokenizer,
    input_messages: List[Dict],
    output_messages: List[Dict],
    max_seq_length: int,
    chat_template_kwargs: Optional[dict] = None,
    assistant_prefill: Optional[str] = None,
) -> dict:
    """-> {"input_ids": [...], "labels": [...], "truncated": bool} with prompt
    tokens labeled -100. Expects exactly one assistant message in
    output_messages. Over-long examples keep the beginning and set truncated
    (a truncated tail loses its EOS supervision — watch the flag counts).

    assistant_prefill is raw text appended to the rendered prompt and labeled
    -100: tokens the model is CONDITIONED on but never trained to produce. Use
    it for a fixed opening of the assistant turn that the chat template cannot
    express. The motivating case is "thinking disabled" SFT on gpt-oss: passing
    "<|channel|>analysis<|message|><|end|><|start|>assistant" puts an empty
    analysis channel in the prompt, mirroring what Qwen3's
    enable_thinking=False does through its template. Supervising that empty
    channel instead would train the model to emit </think> immediately, which
    suppresses reasoning at eval time and destroys any later CoT measurement.
    """
    assistant = [m for m in output_messages if m.get("role") == "assistant"]
    if len(assistant) != 1:
        raise ValueError(
            f"expected exactly 1 assistant message in output, got {len(assistant)}"
        )

    prompt_ids = _as_id_list(tokenizer.apply_chat_template(
        input_messages,
        add_generation_prompt=True,
        tokenize=True,
        **(chat_template_kwargs or {}),
    ))
    if assistant_prefill:
        prompt_ids += list(
            tokenizer(assistant_prefill, add_special_tokens=False)["input_ids"]
        )

    content = assistant[0]["content"]
    if _is_gpt_oss(tokenizer):
        completion_text = _gpt_oss_completion(content)
    else:
        # qwen etc.: plain-string content (typically "<think>\n...\n</think>\n\n
        # ANSWER: X"); the template's generation prompt already opened the
        # assistant turn, so we just append content + EOS (<|im_end|> for qwen).
        if not isinstance(content, str):
            raise ValueError(
                f"expected string assistant content for {tokenizer.name_or_path}, "
                f"got {type(content).__name__}"
            )
        # Qwen3.5/3.6 templates open the think block INSIDE the generation
        # prompt ("<|im_start|>assistant\n<think>\n"), while Qwen3's does not
        # and the stored completions carry their own "<think>\n". Drop the
        # duplicate so we never train on "<think>\n<think>\n".
        think_open = "<think>\n"
        if (content.startswith(think_open)
                and tokenizer.decode(prompt_ids[-4:]).endswith(think_open)):
            content = content[len(think_open):]
        completion_text = content + tokenizer.eos_token
    completion_ids = list(tokenizer(completion_text, add_special_tokens=False)["input_ids"])

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    truncated = len(input_ids) > max_seq_length
    return {
        "input_ids": input_ids[:max_seq_length],
        "labels": labels[:max_seq_length],
        "truncated": truncated,
    }


def doc_end_token_id(tokenizer) -> int:
    """Document separator for raw-text rows. Prefer <|endoftext|> when the vocab
    has it (Qwen and gpt-oss both pretrain with it; their chat EOS —
    <|im_end|> / <|return|> — is a turn terminator, not a document boundary).
    Fall back to eos_token_id."""
    tid = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", None):
        return tid
    if tokenizer.eos_token_id is None:
        raise ValueError(f"{tokenizer.name_or_path}: no <|endoftext|> and no eos token")
    return tokenizer.eos_token_id


def render_text(
    tokenizer,
    text: str,
    max_seq_length: int,
    prefix: Optional[str] = None,
) -> dict:
    """Raw document -> {"input_ids", "labels", "truncated"}: [BOS?] [prefix
    (-100)] text doc_end, with every text token supervised. BOS is added only
    when the tokenizer defines one (Qwen does not; gpt-oss does). Over-long
    documents keep the beginning and lose their doc_end supervision."""
    ids: List[int] = []
    labels: List[int] = []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
        labels.append(tokenizer.bos_token_id)
    if prefix:
        pre = list(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        ids += pre
        labels += [-100] * len(pre)
    body = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    body.append(doc_end_token_id(tokenizer))
    ids += body
    labels += body
    truncated = len(ids) > max_seq_length
    return {
        "input_ids": ids[:max_seq_length],
        "labels": labels[:max_seq_length],
        "truncated": truncated,
    }


def render_row(
    tokenizer,
    row: Dict,
    max_seq_length: int,
    chat_template_kwargs: Optional[dict] = None,
    assistant_prefill: Optional[str] = None,
) -> dict:
    """Dispatch on row format: {"text": ...} -> render_text, else chat.

    A chat row may carry a "render" dict overriding the config-wide
    chat_template_kwargs / assistant_prefill for that row only, e.g.
    {"render": {"chat_template_kwargs": {"reasoning_effort": "medium"},
    "assistant_prefill": null}} to mix reasoning-on rows into a dataset whose
    config conditions reasoning off (kaushik/elicit/nocot_sft). Rows without
    the key render exactly as before."""
    if "text" in row:
        return render_text(tokenizer, row["text"], max_seq_length, prefix=row.get("prefix"))
    if "input" not in row or "output" not in row:
        raise ValueError(f"row needs 'text' or 'input'/'output': keys={list(row)}")
    override = row.get("render") or {}
    return render_example(
        tokenizer, row["input"], row["output"], max_seq_length,
        chat_template_kwargs=override.get("chat_template_kwargs", chat_template_kwargs),
        assistant_prefill=override.get("assistant_prefill", assistant_prefill),
    )


def pack_records(records: List[dict], max_seq_length: int) -> List[dict]:
    """Concatenate rendered records in order into chunks of exactly
    max_seq_length tokens (the last chunk may be shorter). Each record's
    labels stay aligned with its tokens, so masked prefixes/prompts remain
    masked. A record may straddle two chunks; nothing is dropped."""
    ids: List[int] = []
    labels: List[int] = []
    for r in records:
        ids += r["input_ids"]
        labels += r["labels"]
    out = []
    for i in range(0, len(ids), max_seq_length):
        out.append({"input_ids": ids[i:i + max_seq_length],
                    "labels": labels[i:i + max_seq_length]})
    return out
