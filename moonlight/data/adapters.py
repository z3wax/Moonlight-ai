"""Row adapters: raw dataset rows -> plain text / conversations / preference pairs (SPEC §6, §13.1).

A *conversation* is a list of message dicts in the SPEC §5 format::

    {"role": "system"|"user"|"assistant"|"tool", "content": str,
     "tool_calls": [{"name": str, "arguments": dict}],   # assistant, optional
     "images": int,                                      # user, optional
     "name": str}                                        # tool role

Tool schemas of a conversation travel in the leading system message under the extra key
``"tools"`` (list of ``{"name", "description", "parameters"}`` JSON schemas). The tokenizer ignores
unknown keys, and :func:`moonlight.data.sft.encode_conversations` passes them to
``tokenizer.encode_chat(..., tools=...)`` (see :func:`split_tools`).

Public API:
    ``to_text(source_name, row) -> str | None``
    ``to_conversation(source_name, row) -> list[dict] | None``
    ``to_preference(row) -> (chosen_msgs, rejected_msgs) | None``  (also ``to_preference(source, row)``)
    ``to_rl_prompt(source_name, row) -> dict | None``  (GRPO prompt with verifiable answer)
    ``image_captions(source_name, row) -> list[str]``  (vision sources)

Adapters never raise on malformed rows: they return None (the row is skipped). Pure python.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .sources import DatasetSpec, get_source

__all__ = [
    "Conversation", "to_text", "to_conversation", "to_preference", "to_rl_prompt", "image_captions",
    "conversation_to_text", "split_tools", "normalize_conversation", "parse_glaive", "parse_glaive_tools",
    "parse_functioncall", "parse_apigen_calls", "extract_json_values", "gsm8k_final_answer",
    "convert_thoughts", "ADAPTERS",
]

Conversation = List[Dict[str, Any]]

_ROLE_MAP = {
    "system": "system", "user": "user", "human": "user", "prompter": "user", "question": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant", "model": "assistant",
    "chatgpt": "assistant", "answer": "assistant",
    "tool": "tool", "function": "tool", "ipython": "tool", "observation": "tool", "function_response": "tool",
}


# =============================================================================================== utils

def _s(value: Any) -> str:
    """Value -> stripped string ('' for None / NaN-like)."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN from pandas-style rows
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _as_list(value: Any) -> List[Any]:
    """Lists, tuples and numpy arrays -> list; JSON strings decoded; others -> []."""
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        if v[:1] in ("[", "{"):
            try:
                value = json.loads(v)
            except ValueError:
                return []
        else:
            return []
    if isinstance(value, Mapping):
        return [value]
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _msg(role: str, content: Any, **extra: Any) -> Dict[str, Any]:
    m: Dict[str, Any] = {"role": role, "content": _s(content) if not isinstance(content, str) else content.strip()}
    m.update({k: v for k, v in extra.items() if v is not None})
    return m


def normalize_conversation(messages: Sequence[Mapping[str, Any]], require_assistant: bool = True
                           ) -> Optional[Conversation]:
    """Validate / normalize a list of message-like dicts.

    * maps role aliases (human/gpt/from/value ...) to SPEC roles; unknown roles -> conversation dropped;
    * keeps a system message only in first position (later ones are dropped);
    * drops empty messages (assistant tool calls with empty content are kept);
    * merges consecutive user messages;
    * trims trailing non-assistant messages and requires >= 1 user and >= 1 assistant message.
    """
    out: Conversation = []
    for i, raw in enumerate(messages):
        if not isinstance(raw, Mapping):
            return None
        role_raw = raw.get("role", raw.get("from", ""))
        role = _ROLE_MAP.get(_s(role_raw).lower())
        if role is None:
            return None
        content = raw.get("content", raw.get("value", ""))
        content = content if isinstance(content, str) else _s(content)
        content = content.strip()
        m: Dict[str, Any] = {"role": role, "content": content}
        for key in ("tool_calls", "images", "name", "tools"):
            if raw.get(key) not in (None, [], ""):
                m[key] = raw[key]
        if role == "system":
            if out:
                continue
            if not content and "tools" not in m:
                continue
        elif not content and not m.get("tool_calls"):
            continue
        if role == "user" and out and out[-1]["role"] == "user" and not m.get("images"):
            out[-1]["content"] = (out[-1]["content"] + "\n\n" + content).strip()
            continue
        out.append(m)
    while out and out[-1]["role"] != "assistant":
        out.pop()
    if require_assistant:
        if not any(m["role"] == "user" for m in out) or not any(m["role"] == "assistant" for m in out):
            return None
    return out or None


def split_tools(conversation: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[List[Any]]]:
    """Return ``(messages, tools)``: the tool schemas stored in a leading system message's ``"tools"``
    key are removed from the (shallow-copied) messages and returned separately (None if absent)."""
    msgs = [dict(m) for m in conversation]
    tools = None
    if msgs and msgs[0].get("role") == "system" and msgs[0].get("tools"):
        tools = list(msgs[0].pop("tools"))
    return msgs, tools


def conversation_to_text(conversation: Sequence[Mapping[str, Any]]) -> str:
    """Plain-text rendering (for pretraining mixes / tokenizer training), e.g.::

        User: Salom!
        Assistant: Assalomu alaykum! ...

    Tool calls render as ``Assistant (tool call): {json}`` and tool results as ``Tool (name): ...``;
    the system prompt is omitted; ``<|think|>``/``<|/think|>`` markers are kept as text.
    """
    lines = []
    for m in conversation:
        role = m.get("role")
        content = _s(m.get("content"))
        if role == "system":
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            for c in calls:
                lines.append("Assistant (tool call): " + json.dumps(
                    {"name": c.get("name"), "arguments": c.get("arguments", {})}, ensure_ascii=False))
            if content and not calls:
                lines.append(f"Assistant: {content}")
        elif role == "tool":
            name = m.get("name")
            lines.append(f"Tool ({name}): {content}" if name else f"Tool: {content}")
    return "\n".join(lines).strip()


def extract_json_values(text: str, start: int = 0) -> List[Any]:
    """All top-level JSON values (objects/arrays) found in ``text`` from ``start`` (non-JSON skipped)."""
    dec = json.JSONDecoder()
    out: List[Any] = []
    i = start
    n = len(text)
    while i < n:
        j_obj = text.find("{", i)
        j_arr = text.find("[", i)
        cands = [j for j in (j_obj, j_arr) if j >= 0]
        if not cands:
            break
        j = min(cands)
        try:
            val, end = dec.raw_decode(text, j)
        except ValueError:
            i = j + 1
            continue
        out.append(val)
        i = end
    return out


def _schema(obj: Any) -> Optional[Dict[str, Any]]:
    """Normalize one tool definition (plain or OpenAI ``{"type":"function","function":{...}}``)."""
    if not isinstance(obj, Mapping):
        return None
    if isinstance(obj.get("function"), Mapping):
        obj = obj["function"]
    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    params = obj.get("parameters")
    if not isinstance(params, Mapping):
        params = {"type": "object", "properties": {}}
    return {"name": name.strip(), "description": _s(obj.get("description")), "parameters": dict(params)}


def _parse_arguments(raw: Any) -> Optional[Dict[str, Any]]:
    """Tool-call arguments -> dict (accepts dicts, JSON strings, python-literal dict strings)."""
    if isinstance(raw, Mapping):
        return dict(raw)
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    for cand in (s, s.replace("\\'", "'")):
        try:
            val = json.loads(cand)
        except ValueError:
            continue
        if isinstance(val, str):  # double-encoded
            try:
                val = json.loads(val)
            except ValueError:
                continue
        if isinstance(val, Mapping):
            return dict(val)
    try:
        val = ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return dict(val) if isinstance(val, Mapping) else None


def _call(name: Any, arguments: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(name, str) or not name.strip():
        return None
    args = _parse_arguments(arguments)
    if args is None:
        return None
    return {"name": name.strip(), "arguments": args}


# =============================================================================================== glaive

_GLAIVE_TURN_RE = re.compile(r"(?:^|\n)\s*(USER|ASSISTANT|FUNCTION RESPONSE|SYSTEM):", re.MULTILINE)
_ENDOFTEXT = "<|endoftext|>"
_FC_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_FC_ARGS_SQ_RE = re.compile(r'"arguments"\s*:\s*\'(.*)\'\s*}\s*$', re.DOTALL)


def parse_glaive_tools(system: str) -> List[Dict[str, Any]]:
    """Tool schemas from a glaive system string (``SYSTEM: ... functions ... -\\n{json}\\n\\n{json}``)."""
    text = system or ""
    if "no access to external functions" in text.lower():
        return []
    m = re.search(r"functions?[^\n]*?-\s*\n", text)
    start = m.end() if m else 0
    out = []
    for val in extract_json_values(text, start):
        items = val if isinstance(val, list) else [val]
        for it in items:
            sch = _schema(it)
            if sch is not None:
                out.append(sch)
    return out


def parse_functioncall(text: str) -> Optional[Dict[str, Any]]:
    """Parse ``<functioncall> {"name": "f", "arguments": '{"a": 1}'}`` (arguments is a JSON string in
    single quotes in glaive v2; plain dicts are accepted too). Returns ``{"name", "arguments"}``."""
    idx = text.find("<functioncall>")
    body = (text[idx + len("<functioncall>"):] if idx >= 0 else text).replace(_ENDOFTEXT, "").strip()
    if not body:
        return None
    # 1) valid JSON already (arguments as dict or as a double-quoted string)
    vals = extract_json_values(body)
    for val in vals:
        if isinstance(val, Mapping) and "name" in val:
            call = _call(val.get("name"), val.get("arguments", {}))
            if call is not None:
                return call
    # 2) glaive style: "arguments": '{...}'
    name_m = _FC_NAME_RE.search(body)
    args_m = _FC_ARGS_SQ_RE.search(body)
    if name_m and args_m:
        return _call(name_m.group(1), args_m.group(1))
    if name_m and '"arguments"' not in body:
        return _call(name_m.group(1), {})
    return None


def parse_glaive(system: str, chat: str) -> Optional[Conversation]:
    """Convert one glaive-function-calling-v2 example to a Moonlight conversation.

    ``ASSISTANT: <functioncall> {...}`` -> assistant message with ``tool_calls``; ``FUNCTION RESPONSE:``
    -> ``tool`` message (``name`` = the preceding call); ``<|endoftext|>`` is stripped. The available
    tools are stored in the system message's ``"tools"`` key (default system prompt text). Returns
    None when a function call cannot be parsed (broken calls would teach a broken format).
    """
    from ..tokenizer import DEFAULT_SYSTEM_PROMPT

    tools = parse_glaive_tools(system or "")
    text = (chat or "").replace(_ENDOFTEXT, "")
    marks = list(_GLAIVE_TURN_RE.finditer(text))
    if not marks:
        return None
    msgs: Conversation = []
    sys_msg: Dict[str, Any] = {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    if tools:
        sys_msg["tools"] = tools
    msgs.append(sys_msg)
    last_call: Optional[str] = None
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        role = m.group(1)
        content = text[m.end():end].strip()
        if role == "USER":
            if content:
                msgs.append(_msg("user", content))
        elif role == "ASSISTANT":
            if "<functioncall>" in content:
                call = parse_functioncall(content)
                if call is None:
                    return None
                last_call = call["name"]
                prefix = content[:content.find("<functioncall>")].strip()
                if prefix:
                    msgs.append(_msg("assistant", prefix))
                msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
            elif content:
                msgs.append(_msg("assistant", content))
        elif role == "FUNCTION RESPONSE":
            msgs.append(_msg("tool", content, name=last_call))
    return normalize_conversation(msgs)


# =============================================================================================== apigen

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOLS_TAG_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)


def parse_apigen_calls(content: str) -> Optional[List[Dict[str, Any]]]:
    """Tool calls in an assistant message: ``<tool_call>[...]</tool_call>`` / ``<tool_call>{...}`` or a
    message that is entirely a JSON list/object of ``{"name", "arguments"}``. None if not a call."""
    text = (content or "").strip()
    if not text:
        return None
    tagged = _TOOL_CALL_TAG_RE.findall(text)
    if tagged:
        payloads = tagged
    elif text[:1] in ("[", "{") and text[-1:] in ("]", "}"):
        payloads = [text]
    else:
        return None
    calls: List[Dict[str, Any]] = []
    for p in payloads:
        vals = extract_json_values(p)
        if not vals:
            return None
        for val in vals:
            for item in (val if isinstance(val, list) else [val]):
                if not isinstance(item, Mapping):
                    return None
                fn = item.get("function") if isinstance(item.get("function"), Mapping) else item
                call = _call(fn.get("name"), fn.get("arguments", fn.get("parameters", {})))
                if call is None:
                    return None
                calls.append(call)
    return calls or None


def _apigen_tools(system: str) -> List[Dict[str, Any]]:
    m = _TOOLS_TAG_RE.search(system or "")
    vals = extract_json_values(m.group(1)) if m else extract_json_values(system or "")
    out = []
    for val in vals:
        for it in (val if isinstance(val, list) else [val]):
            sch = _schema(it)
            if sch is not None:
                out.append(sch)
    return out


# =============================================================================================== row -> conv

def _conv_messages(row: Mapping[str, Any]) -> Optional[Conversation]:
    msgs = _as_list(row.get("messages"))
    return normalize_conversation(msgs) if msgs else None


def _conv_apigen(row: Mapping[str, Any]) -> Optional[Conversation]:
    """smoltalk apigen-80k: tool definitions in the system prompt, calls in assistant content (kept as
    text when they cannot be parsed)."""
    from ..tokenizer import DEFAULT_SYSTEM_PROMPT

    msgs = _as_list(row.get("messages"))
    if not msgs:
        return None
    out: List[Dict[str, Any]] = []
    tools: List[Dict[str, Any]] = []
    for raw in msgs:
        if not isinstance(raw, Mapping):
            return None
        role = _ROLE_MAP.get(_s(raw.get("role")).lower())
        content = _s(raw.get("content"))
        if role == "system":
            tools = _apigen_tools(content)
            if not tools:
                out.append(_msg("system", content))
            continue
        if role == "assistant":
            calls = parse_apigen_calls(content)
            if calls:
                out.append({"role": "assistant", "content": "", "tool_calls": calls})
                continue
        if role is None:
            return None
        out.append(_msg(role, content))
    if tools:
        out.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT, "tools": tools})
    return normalize_conversation(out)


def _conv_alpaca(row: Mapping[str, Any]) -> Optional[Conversation]:
    instr, inp, out = _s(row.get("instruction")), _s(row.get("input")), _s(row.get("output"))
    if not instr or not out:
        return None
    user = f"{instr}\n\n{inp}" if inp else instr
    return normalize_conversation([_msg("user", user), _msg("assistant", out)])


def _conv_query_answer(row: Mapping[str, Any]) -> Optional[Conversation]:
    q, a = _s(row.get("query")), _s(row.get("answer"))
    if not q or not a:
        return None
    return normalize_conversation([_msg("user", q), _msg("assistant", a)])


def _strip_quotes(text: str) -> str:
    """Strip one pair of outer quotes and unescape ``\\"`` (SFT-UZ-9k artifacts)."""
    t = (text or "").strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t.replace('\\"', '"').replace("\\n", "\n").strip()


def _conv_sft_uz_9k(row: Mapping[str, Any]) -> Optional[Conversation]:
    prompt, answer = _strip_quotes(_s(row.get("prompt"))), _strip_quotes(_s(row.get("answer")))
    if not prompt or not answer:
        return None
    msgs: Conversation = []
    system = _strip_quotes(_s(row.get("system")))
    if system:
        msgs.append(_msg("system", system))
    for turn in _as_list(row.get("history")):
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, a = _strip_quotes(_s(turn[0])), _strip_quotes(_s(turn[1]))
            if u and a:
                msgs += [_msg("user", u), _msg("assistant", a)]
        elif isinstance(turn, Mapping):
            role = _ROLE_MAP.get(_s(turn.get("role", turn.get("from"))).lower())
            if role in ("user", "assistant"):
                msgs.append(_msg(role, _strip_quotes(_s(turn.get("content", turn.get("value"))))))
    msgs += [_msg("user", prompt), _msg("assistant", answer)]
    return normalize_conversation(msgs)


_LLAMA3_RE = re.compile(
    r"<\|start_header_id\|>\s*(\w+)\s*<\|end_header_id\|>\s*(.*?)\s*(?=<\|eot_id\|>|<\|start_header_id\|>|<\|end_of_text\|>|$)",
    re.DOTALL)


def _conv_llama3(row: Mapping[str, Any]) -> Optional[Conversation]:
    text = _s(row.get("text"))
    if "<|start_header_id|>" not in text:
        return None
    msgs = [{"role": role.lower(), "content": content} for role, content in _LLAMA3_RE.findall(text)]
    return normalize_conversation(msgs) if msgs else None


def _conv_sharegpt(row: Mapping[str, Any]) -> Optional[Conversation]:
    turns = _as_list(row.get("conversations"))
    return normalize_conversation(turns) if turns else None


_THOUGHT_RE = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)


def convert_thoughts(text: str) -> str:
    """OpenThoughts markers -> Moonlight: ``<|begin_of_thought|>X<|end_of_thought|>`` ->
    ``<|think|>X<|/think|>``; ``<|begin_of_solution|>``/``<|end_of_solution|>`` removed."""
    def repl(m: "re.Match[str]") -> str:
        return "<|think|>" + m.group(1).strip() + "<|/think|>"

    out = _THOUGHT_RE.sub(repl, text or "")
    out = out.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "")
    out = re.sub(r"<\|/think\|>\s+", "<|/think|>", out)
    return out.strip()


def _conv_openthoughts(row: Mapping[str, Any]) -> Optional[Conversation]:
    """OpenThoughts-114k. Its long English "think step by step" system prompt is dropped (the default
    Moonlight system prompt is used) because the reasoning format is already in the target."""
    turns = _as_list(row.get("conversations"))
    msgs = []
    for t in turns:
        if not isinstance(t, Mapping):
            return None
        role = _ROLE_MAP.get(_s(t.get("from", t.get("role"))).lower())
        content = _s(t.get("value", t.get("content")))
        if role == "assistant":
            if "<|begin_of_thought|>" in content and "<|end_of_thought|>" not in content:
                return None  # truncated reasoning
            content = convert_thoughts(content)
        if role is None:
            return None
        msgs.append({"role": role, "content": content})
    return normalize_conversation(msgs)


_GSM_CALC_RE = re.compile(r"<<[^<>]*>>")


def gsm8k_final_answer(answer: str) -> Optional[str]:
    """The number after ``####`` in a GSM8K answer (commas removed), or None."""
    m = re.search(r"####\s*([-+]?[\d,]*\.?\d+)", answer or "")
    return m.group(1).replace(",", "") if m else None


def _conv_gsm8k(row: Mapping[str, Any]) -> Optional[Conversation]:
    """GSM8K as reasoning SFT: ``<|think|>steps<|/think|>`` + ``Answer: N`` (calculator annotations
    ``<<a*b=c>>`` removed)."""
    q, a = _s(row.get("question")), _s(row.get("answer"))
    final = gsm8k_final_answer(a)
    if not q or final is None:
        return None
    steps = _GSM_CALC_RE.sub("", a.split("####")[0]).strip()
    return normalize_conversation([_msg("user", q),
                                   _msg("assistant", f"<|think|>{steps}<|/think|>Answer: {final}")])


_UZ_CAPTION_PROMPTS = ("Rasmda nima tasvirlangan?", "Ushbu rasmni qisqacha tasvirlab bering.",
                       "Rasmni tasvirlab ber.")
_EN_CAPTION_PROMPTS = ("Describe this image.", "What is shown in the picture?",
                       "Write a short caption for this image.")


def image_captions(source_name: str, row: Mapping[str, Any]) -> List[str]:
    """Non-empty captions of a vision row (flickr8k: ``caption_0..4``; flickr30k: ``caption`` list)."""
    caps: List[str] = []
    if "caption" in row:
        val = row.get("caption")
        caps = [_s(c) for c in (_as_list(val) if not isinstance(val, str) else [val])]
    for i in range(10):
        key = f"caption_{i}"
        if key in row:
            caps.append(_s(row.get(key)))
    return [c for c in caps if c]


def _conv_caption(row: Mapping[str, Any], source: str = "") -> Optional[Conversation]:
    caps = image_captions(source, row)
    if not caps:
        return None
    return [_msg("user", _EN_CAPTION_PROMPTS[0], images=1), _msg("assistant", caps[0])]


def _identity_conv(row: Mapping[str, Any]) -> Optional[Conversation]:
    return None


# =============================================================================================== row -> text

def _text_text(row: Mapping[str, Any]) -> Optional[str]:
    text = _s(row.get("text"))
    title = _s(row.get("title"))
    if not text:
        return None
    return f"{title}\n\n{text}" if title and not text.startswith(title) else text


def _text_content(row: Mapping[str, Any]) -> Optional[str]:
    content = row.get("content")
    return content if isinstance(content, str) and content.strip() else None


# =============================================================================================== tables

#: adapter name -> (row -> text, row -> conversation). A None text function renders the conversation.
ADAPTERS: Dict[str, Tuple[Optional[Callable[[Mapping[str, Any]], Optional[str]]],
                          Optional[Callable[[Mapping[str, Any]], Optional[Conversation]]]]] = {
    "text": (_text_text, None),
    "content": (_text_content, None),
    "alpaca": (None, _conv_alpaca),
    "messages": (None, _conv_messages),
    "apigen": (None, _conv_apigen),
    "sft_uz_9k": (None, _conv_sft_uz_9k),
    "llama3_text": (None, _conv_llama3),
    "sharegpt": (None, _conv_sharegpt),
    "openthoughts": (None, _conv_openthoughts),
    "query_answer": (None, _conv_query_answer),
    "glaive": (None, lambda r: parse_glaive(_s(r.get("system")), _s(r.get("chat")))),
    "preference": (None, lambda r: (to_preference(r) or (None, None))[0]),
    "gsm8k": (None, _conv_gsm8k),
    "flickr8k": (None, _conv_caption),
    "flickr30k": (None, _conv_caption),
    "identity": (None, _identity_conv),
}


def _adapter(source_name: str) -> Tuple[DatasetSpec, Any, Any]:
    spec = get_source(source_name)
    text_fn, conv_fn = ADAPTERS[spec.adapter]
    return spec, text_fn, conv_fn


def to_text(source_name: str, row: Mapping[str, Any]) -> Optional[str]:
    """Raw row -> document text (pretraining). Conversation sources are rendered with
    :func:`conversation_to_text`; benchmark sources return None. Never raises on bad rows."""
    spec, text_fn, conv_fn = _adapter(source_name)
    try:
        if text_fn is not None:
            return text_fn(row)
        if conv_fn is not None and spec.kind != "bench":
            conv = conv_fn(row)
            return conversation_to_text(conv) if conv else None
    except (TypeError, ValueError, AttributeError, KeyError, IndexError):
        return None
    return None


def to_conversation(source_name: str, row: Mapping[str, Any]) -> Optional[Conversation]:
    """Raw row -> conversation (list of SPEC §5 messages) or None (row skipped)."""
    spec, text_fn, conv_fn = _adapter(source_name)
    if conv_fn is None:
        return None
    try:
        return conv_fn(row)
    except (TypeError, ValueError, AttributeError, KeyError, IndexError, RecursionError):
        return None


def to_preference(row_or_source: Any, row: Optional[Mapping[str, Any]] = None
                  ) -> Optional[Tuple[Conversation, Conversation]]:
    """UltraFeedback-binarized row -> ``(chosen_msgs, rejected_msgs)`` or None.

    Accepts ``to_preference(row)`` (SPEC) and ``to_preference(source_name, row)``. Both conversations
    must share the prompt, end with a (different) assistant answer and ``score_chosen`` must exceed
    ``score_rejected`` when both scores are present.
    """
    if row is None:
        row = row_or_source
    if not isinstance(row, Mapping):
        return None
    try:
        sc, sr = row.get("score_chosen"), row.get("score_rejected")
        if sc is not None and sr is not None and float(sc) <= float(sr):
            return None
        chosen = normalize_conversation(_as_list(row.get("chosen")))
        rejected = normalize_conversation(_as_list(row.get("rejected")))
    except (TypeError, ValueError):
        return None
    if not chosen or not rejected:
        prompt = _s(row.get("prompt"))
        c_txt = _s(row.get("chosen")) if isinstance(row.get("chosen"), str) else ""
        r_txt = _s(row.get("rejected")) if isinstance(row.get("rejected"), str) else ""
        if not (prompt and c_txt and r_txt):
            return None
        chosen = [_msg("user", prompt), _msg("assistant", c_txt)]
        rejected = [_msg("user", prompt), _msg("assistant", r_txt)]
    if chosen[:-1] != rejected[:-1] or chosen[-1]["content"] == rejected[-1]["content"]:
        return None
    return chosen, rejected


def to_rl_prompt(source_name: str, row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """GRPO prompt from a source row: ``{"prompt_messages", "reward_type", "reward_data"}``.

    Supported: ``gsm8k-train`` / ``gsm8k`` (reward ``exact_number`` with ``{"target": float}``).
    """
    spec = get_source(source_name)
    if spec.adapter == "gsm8k" or spec.name in ("gsm8k", "gsm8k-train"):
        q = _s(row.get("question"))
        final = gsm8k_final_answer(_s(row.get("answer")))
        if not q or final is None:
            return None
        return {"prompt_messages": [_msg("user", q)], "reward_type": "exact_number",
                "reward_data": {"target": float(final)}}
    return None
