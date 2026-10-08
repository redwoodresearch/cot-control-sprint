"""Text tool-call protocol for the minimal agent loop.

The canonical format is Qwen3's native one, which small models emit reliably
and which gpt-oss follows once shown an example:

    <tool_call>
    {"name": "read_file", "arguments": {"path": "grader.py"}}
    </tool_call>

`parse_tool_calls` is deliberately forgiving. It also accepts unclosed or
variant tags, markdown code fences, python call syntax (``read_file("x")``),
single-quoted / trailing-comma JSON, a bare JSON object, ``arguments`` given
as a JSON string or under another key, and gpt-oss harmony commentary
channels (``<|channel|>commentary to=functions.read_file<|message|>{...}``)
in the raw completion. Unknown tool names are kept (the loop reports them to
the model) unless they fuzzy-match a known tool.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str = ""
    known: bool = True

    def render(self) -> str:
        body = json.dumps({"name": self.name, "arguments": self.arguments}, ensure_ascii=False)
        return f"<tool_call>\n{body}\n</tool_call>"


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict = field(default_factory=dict)  # arg name -> description

    def signature(self) -> str:
        args = ", ".join(f"{k}: str" for k in self.params)
        return f"{self.name}({args})"


_TAG = r"(?:tool_call|tool-call|toolcall|function_call|function|tool|call)"
_BLOCK_RE = re.compile(rf"<\s*{_TAG}\s*>(.*?)(?:<\s*/\s*{_TAG}\s*>|\Z)", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|python|py|tool_call)?\s*(.*?)```", re.DOTALL)
# A JSON object with one nested level, containing a "name" key.
_JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\"name\"(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)
# A harmony commentary segment: header (may carry to=<tool>, may or may not end
# in <|message|>; gpt-oss-20b also emits "to=tool_call>{...}" and
# "to=tool_call name=read_file arguments='{...}'>") followed by the body.
_HARMONY_RE = re.compile(
    r"<\|channel\|>\s*(?:commentary|analysis(?=\s+to=))(?P<header>[^{\n]*?)(?:<\|message\|>|(?<!\|)>|(?=\{))(?P<body>.*?)"
    r"(?=<\|call\||<\|end\||<\|return\||<\|channel\||<\|start\||\Z)",
    re.DOTALL,
)
_HARMONY_TO_RE = re.compile(r"to=([\w.:\-]+)")
_HARMONY_NAME_RE = re.compile(r"\bname=([\w.]+)")
_HARMONY_ARGS_RE = re.compile(r"arguments=(['\"])(.*?)\1", re.DOTALL)
_ANY_OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)
_ARG_KEYS = ("arguments", "args", "parameters", "params", "input", "inputs")


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1) if m and text.strip().startswith("```") else text


def _loads_lenient(text: str) -> dict | None:
    """json.loads with fallbacks for python-dict syntax and trailing commas."""
    text = _strip_fences(text).strip()
    if not text:
        return None
    if text.startswith("{{") and text.endswith("}}"):  # doubled braces: a common model slip
        text = text[1:-1].strip()
    for candidate in (text, re.sub(r",\s*([}\]])", r"\1", text)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except Exception:  # noqa: BLE001 - e.g. TypeError from a set literal holding a dict ({{...}})
        pass
    m = _JSON_OBJ_RE.search(text)
    if m and m.group(0) != text:
        return _loads_lenient(m.group(0))
    return None


ALIASES = {  # names models invent for the three tools
    "list_files": ("ls", "list", "listdir", "list_dir", "list_directory", "dir", "tree", "find", "glob"),
    "read_file": ("open_file", "open", "view_file", "view", "read", "cat", "get_file", "show_file", "print_file",
                  "read_text", "load_file", "file_read"),
    "submit_answer": ("submit", "answer", "final_answer", "finish", "done", "final", "submit_final_answer"),
}
SHELL_NAMES = ("exec", "bash", "shell", "sh", "run", "run_shell", "terminal", "execute", "command", "cmd",
               "run_command", "container_exec", "python")


def _match_name(name: str, known: Sequence[str]) -> tuple[str, bool]:
    name = str(name).strip()
    name = name.split(".")[-1]  # functions.read_file -> read_file
    squash = lambda s: re.sub(r"[_\-\s]", "", s.lower())  # noqa: E731
    low = squash(name)
    for canon, aliases in ALIASES.items():
        if canon in known and low in {squash(a) for a in aliases}:
            return canon, True
    for k in known:
        if low == squash(k):
            return k, True
    for k in known:
        if squash(k) in low or (len(low) >= 4 and low in squash(k)):
            return k, True
    return name, False


def _from_dict(obj: dict, known: Sequence[str], raw: str) -> ToolCall | None:
    name = obj.get("name") or obj.get("tool") or obj.get("function") or obj.get("tool_name")
    if isinstance(name, dict):  # OpenAI-style {"function": {"name":..., "arguments":...}}
        return _from_dict({**obj, **name, "function": None}, known, raw)
    if not name or not isinstance(name, str):
        return None
    args = None
    for k in _ARG_KEYS:
        if k in obj:
            args = obj[k]
            break
    if isinstance(args, str):
        parsed = _loads_lenient(args)
        args = parsed if parsed is not None else ({"path": args} if args.strip() else {})
    # provider quirk (DekaLLM): arguments = '{"name": "list_files", "arguments": {}}' -> unwrap
    if isinstance(args, dict) and "arguments" in args and set(args) <= {"name", "arguments"}:
        args = args["arguments"] if isinstance(args["arguments"], dict) else {}
    if args is None:
        args = {k: v for k, v in obj.items()
                if k not in ("name", "tool", "function", "tool_name", "id", "type") and v is not None}
    if not isinstance(args, dict):
        args = {}
    name, is_known = _match_name(name, known)
    return ToolCall(name=name, arguments={str(k): v for k, v in args.items()}, raw=raw, known=is_known)


def _from_pycall(text: str, known: Sequence[str]) -> list[ToolCall]:
    calls = []
    if not known:
        return calls
    names = "|".join(re.escape(k) for k in known)
    for m in re.finditer(rf"\b(?:functions\.)?({names})\s*\((.*?)\)", text, re.DOTALL):
        name, inner = m.group(1), m.group(2).strip()
        args: dict = {}
        if inner:
            try:
                node = ast.parse(f"f({inner})", mode="eval").body
                pos = [ast.literal_eval(a) for a in node.args]
                kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
                args = dict(kw)
                if pos:
                    args["__positional__"] = pos
            except Exception:  # noqa: BLE001
                s = re.search(r"[\"']([^\"']*)[\"']", inner)
                if s:
                    args = {"__positional__": [s.group(1)]}
        calls.append(ToolCall(name=name, arguments=args, raw=m.group(0), known=True))
    return calls


def structured_tool_calls(raw: object, known: Sequence[str] = ()) -> list[ToolCall]:
    """Native tool calls from an OpenAI-style response dict (choices[0].message.tool_calls)."""
    if not isinstance(raw, dict):
        return []
    try:
        tcs = raw["choices"][0]["message"].get("tool_calls") or []
    except (KeyError, IndexError, TypeError, AttributeError):
        return []
    out = []
    for tc in tcs:
        fn = (tc or {}).get("function") or tc
        obj = {"name": fn.get("name"), "arguments": fn.get("arguments")}
        parsed = _from_dict(obj, known, json.dumps(tc, ensure_ascii=False)) if obj["name"] else None
        if parsed is not None:
            out.append(parsed)
    return out


def parse_tool_calls(
    text: str | None, known: Sequence[str] = (), raw: str | dict | None = None
) -> list[ToolCall]:
    """Extract tool calls from a model message: native tool_calls in an
    OpenAI-style response dict first, then the visible content, then the raw
    completion text (harmony commentary channels). Returns calls in order of
    appearance; may be empty."""
    calls: list[ToolCall] = structured_tool_calls(raw, known)
    if calls:
        return calls
    text = text or ""
    for m in _BLOCK_RE.finditer(text):
        obj = _loads_lenient(m.group(1))
        tc = _from_dict(obj, known, m.group(0)) if obj else None
        if tc is None:
            tc = next(iter(_from_pycall(m.group(1), known)), None)
        if tc is not None:
            calls.append(tc)
    if not calls:
        for m in _FENCE_RE.finditer(text):
            obj = _loads_lenient(m.group(1))
            tc = _from_dict(obj, known, m.group(0)) if obj else None
            if tc is not None:
                calls.append(tc)
    if not calls:
        for m in _JSON_OBJ_RE.finditer(text):
            obj = _loads_lenient(m.group(0))
            tc = _from_dict(obj, known, m.group(0)) if obj else None
            if tc is not None:
                calls.append(tc)
    if not calls:
        calls = _from_pycall(text, known)
    if not calls and isinstance(raw, str) and "<|channel|>" in raw:
        for m in _HARMONY_RE.finditer(raw):
            header, body = m.group("header"), m.group("body").strip()
            body = re.sub(r"</?commentary>", "", body).strip()
            obj = _loads_lenient(body) or {}
            if not obj:
                j = _JSON_OBJ_RE.search(body) or _ANY_OBJ_RE.search(body)
                obj = (_loads_lenient(j.group(0)) if j else None) or {}
            # names from the header: to=functions.read_file, to=tool_call name=read_file, to=tool:read_file
            hint = None
            for rx in (_HARMONY_NAME_RE, _HARMONY_TO_RE):
                h = rx.search(header)
                if h:
                    cand, is_known = _match_name(h.group(1).replace(":", "."), known)
                    if is_known:
                        hint = cand
                        break
            if "name" not in obj:
                a = _HARMONY_ARGS_RE.search(header)
                args = _loads_lenient(a.group(2)) if a else None
                to = _HARMONY_TO_RE.search(header)
                if hint and (args or obj or not body):
                    obj = {"name": hint, "arguments": args or obj}
                elif to and (args or obj):  # unknown target (e.g. container.exec): keep it, the loop may map it
                    obj = {"name": to.group(1).split(".")[-1].split(":")[-1], "arguments": args or obj}
                elif args or obj:
                    continue
            tc = _from_dict(obj, known, m.group(0)) if obj else None
            if tc is None and hint:
                tc = next(iter(_from_pycall(body, known)), None)
            if tc is not None:
                calls.append(tc)
    return calls


def parse_tagged_calls(text: str | None, known: Sequence[str] = ()) -> list[ToolCall]:
    """Strict variant: only explicit <tool_call>{json}</tool_call> blocks (used on the reasoning
    text, where gpt-oss sometimes drafts its calls before a confabulated visible reply)."""
    calls = []
    for m in _BLOCK_RE.finditer(text or ""):
        obj = _loads_lenient(m.group(1))
        tc = _from_dict(obj, known, m.group(0)) if obj else None
        if tc is not None and tc.known:
            calls.append(tc)
    return calls


def openai_tools(tools: Iterable[ToolSpec]) -> list[dict]:
    """OpenAI function-calling schema for `tools` (for API backends that parse tool calls natively)."""
    return [{
        "type": "function",
        "function": {
            "name": t.name, "description": t.description,
            "parameters": {"type": "object",
                           "properties": {k: {"type": "string", "description": v} for k, v in t.params.items()},
                           "required": list(t.params)},
        },
    } for t in tools]


def render_system_prompt(tools: Iterable[ToolSpec], preamble: str) -> str:
    tool_lines = []
    for t in tools:
        params = "; ".join(f"{k}: {v}" for k, v in t.params.items()) or "no arguments"
        tool_lines.append(f"- {t.signature()} -- {t.description} ({params})")
    example = ToolCall("read_file", {"path": "README.md"}).render()
    return (
        f"{preamble.rstrip()}\n\n"
        "You can only act by calling tools. Available tools:\n"
        + "\n".join(tool_lines)
        + "\n\nCall a tool by writing, in your reply, exactly this format (one JSON object per call; "
        "you may include several calls in one reply):\n"
        f"{example}\n\n"
        "The environment replies with the tool results. Finish by calling submit_answer exactly "
        "once; nothing after that is read."
    )
