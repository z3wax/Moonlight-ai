"""Multi-turn chat with tool calling and images (SPEC §8) + planning / project building (§13.6).

```python
session = ChatSession(llm, tools=[calculator])     # or tools=True for the built-in tools
session.send("12 * 7 nechaga teng?")              # -> str (tool loop runs transparently)
session.send("Bu rasmda nima bor?", images=["cat.jpg"])   # multimodal models only
session.history                                    # SPEC §5 messages (user/assistant/tool)
session.reset()
```

Tool loop (per :meth:`ChatSession.send`): the conversation is rendered with the SPEC §5 template
(``<|tools|>`` schemas in the system block) and the model generates until ``<|end|>`` / ``<|eos|>``.
If the FIRST generated token is ``<|tool_call|>`` (detected by token id) the rest is parsed with
:func:`moonlight.tools.parse_tool_call`, executed through the :class:`~moonlight.tools.ToolRegistry`
and appended as ``{"role": "assistant", "tool_calls": [call]}`` + ``{"role": "tool", "name": ...,
"content": result}``; then the model generates again. After ``max_tool_rounds`` calls the
``<|tool_call|>`` token is banned so the model has to answer. A tool call that cannot be parsed
is treated as a plain answer (no crash on random / undertrained models).

This module is torch-free at import time (torch is used lazily for image features only).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from moonlight.tokenizer import (
    ASSISTANT_ID,
    AUDIO_ID,
    BOS_ID,
    END_ID,
    EOS_ID,
    IMAGE_ID,
    NUM_SPECIAL_TOKENS,
    PAD_ID,
    SYSTEM_ID,
    THINK_END_ID,
    THINK_ID,
    TOOL_CALL_ID,
    TOOL_RESULT_ID,
    TOOLS_ID,
    USER_ID,
)
from moonlight.tools import ToolRegistry, Workspace, as_registry, parse_tool_call, workspace_tools

__all__ = [
    "ChatSession",
    "ProjectResult",
    "build_project",
    "PLAN_SYSTEM_PROMPT",
    "BUILDER_SYSTEM_PROMPT",
    "BUILD_NUDGE",
    "GEN_KEYS",
    "resolve_generation",
    "banned_special_ids",
    "split_reply",
]

# -------------------------------------------------------------------------------------------------
# System prompts (§13.6): one Uzbek + one English sentence each.
# -------------------------------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = (
    "Sen Moonlight — puxta rejalashtiruvchi yordamchisan: avval <|think|> ichida maqsad, cheklovlar, "
    "noma'lumlar va xavflarni qisqa tahlil qil, so'ng **Reja / Plan** bo'limida raqamlangan qadamlarni "
    "(har birida harakat, kutilgan natija va tekshiruv) yoz va oxirida **Tekshiruv / Verification** "
    "ro'yxatida foydalanuvchining har bir talabini uni bajaradigan qadamga bog'la. "
    "You are Moonlight, a careful planner: briefly analyse the goal, constraints, unknowns and risks, "
    "then give a numbered Plan (action, expected output and check for every step) and a Verification "
    "checklist that maps every user requirement to the step that satisfies it."
)

BUILDER_SYSTEM_PROMPT = (
    "Sen Moonlight — kichik dasturiy loyihalarni bosqichma-bosqich quradigan agentsan: avval reja tuz, "
    "so'ng write_file, read_file, list_dir va run_tests vositalari yordamida kod va testlarni yoz, "
    "testlarni ishga tushir va hammasi o'tguncha xatolarni tuzat, oxirida har bir talab qayerda "
    "bajarilganini sanab qisqa xulosa ber. "
    "You are Moonlight, an agent that builds small software projects: plan first, write the code and "
    "its tests with the workspace tools, run the tests and fix failures until they pass, then give a "
    "final summary listing every requirement and where it is satisfied."
)

#: follow-up user message used by :func:`build_project` while the tests are missing / failing.
BUILD_NUDGE = (
    "Testlar hali yozilmagan yoki o'tmayapti: testlarni yoz, run_tests bilan ishga tushir va xatolarni "
    "tuzat. Tests are missing or failing: write the tests, run them with run_tests and fix the code."
)

# -------------------------------------------------------------------------------------------------
# Generation settings
# -------------------------------------------------------------------------------------------------

#: generation keyword arguments accepted by chat / generate APIs. ``best_of`` and ``n`` are aliases
#: (number of samples for best-of-N); ``stream`` is a ``callable(text_delta: str)``;
#: ``show_thinking`` keeps the ``<|think|>`` part in returned replies; ``banned_ids`` adds ids that
#: must never be generated.
GEN_KEYS = ("max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty", "no_repeat_ngram",
            "best_of", "n", "seed", "stream", "show_thinking", "banned_ids")


def resolve_generation(chat_cfg: Any, *overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge ``config.chat`` defaults with override dicts (later wins; None values are ignored).

    Returns a dict with every key of :data:`GEN_KEYS` (``n`` resolved from ``best_of``/``n``).
    Unknown keys raise ``TypeError``. Sampling with ``temperature <= 0`` (greedy) or a ``stream``
    callback forces ``n = 1``.
    """
    g: Dict[str, Any] = {
        "max_new_tokens": int(getattr(chat_cfg, "max_new_tokens", 256)),
        "temperature": float(getattr(chat_cfg, "temperature", 0.7)),
        "top_k": int(getattr(chat_cfg, "top_k", 40)),
        "top_p": float(getattr(chat_cfg, "top_p", 0.9)),
        "repetition_penalty": float(getattr(chat_cfg, "repetition_penalty", 1.1)),
        "no_repeat_ngram": int(getattr(chat_cfg, "no_repeat_ngram", 0)),
        "n": int(getattr(chat_cfg, "best_of", 1)),
        "seed": None,
        "stream": None,
        "show_thinking": False,
        "banned_ids": (),
    }
    for ov in overrides:
        for k, v in (ov or {}).items():
            if k not in GEN_KEYS:
                raise TypeError(f"unknown generation argument {k!r}; expected one of: {', '.join(GEN_KEYS)}")
            if v is None:
                continue
            g["n" if k == "best_of" else k] = v
    g["n"] = max(1, int(g["n"]))
    g["max_new_tokens"] = max(1, int(g["max_new_tokens"]))
    if float(g["temperature"]) <= 0 or g["stream"] is not None:
        g["n"] = 1
    if g["stream"] is not None and not callable(g["stream"]):
        raise TypeError("stream must be a callable receiving text deltas")
    return g


def banned_special_ids(tokenizer_vocab: int, model_vocab: int, allow_tool_call: bool) -> List[int]:
    """Ids a reply must never contain: structural specials (pad, bos, role markers, image, tool
    result/tools block, audio, reserved), ``<|tool_call|>`` unless tools are enabled, and ids the
    tokenizer cannot decode (``tokenizer_vocab <= id < model_vocab``). ``<|end|>`` / ``<|eos|>``
    stay allowed (they stop generation); ``<|think|>``, ``<|/think|>``, ``<|code|>`` are allowed."""
    banned = {PAD_ID, BOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID, IMAGE_ID, TOOL_RESULT_ID, TOOLS_ID, AUDIO_ID}
    banned.update(range(15, NUM_SPECIAL_TOKENS))
    if not allow_tool_call:
        banned.add(TOOL_CALL_ID)
    banned.update(range(max(int(tokenizer_vocab), NUM_SPECIAL_TOKENS), int(model_vocab)))
    return sorted(i for i in banned if 0 <= i < int(model_vocab))


def split_reply(tokenizer: Any, tokens: Sequence[int], show_thinking: bool = False) -> Tuple[str, str]:
    """Generated reply tokens (stop token excluded) -> ``(history_content, visible_text)``.

    ``history_content`` keeps a ``<|think|>…<|/think|>`` block (as special-token strings, which the
    chat template re-encodes as the special ids) so the conversation round-trips; ``visible_text``
    is the stripped answer after the thinking block (or the whole text with the block when
    ``show_thinking``). A thinking block that never closes yields its text as the answer.
    """
    toks = [int(t) for t in tokens]
    if THINK_ID not in toks:
        text = tokenizer.decode(toks, skip_special=True)
        return text, text.strip()
    i = toks.index(THINK_ID)
    before = tokenizer.decode(toks[:i], skip_special=True)
    rest = toks[i + 1:]
    if THINK_END_ID in rest:
        j = rest.index(THINK_END_ID)
        think = tokenizer.decode(rest[:j], skip_special=True)
        answer = tokenizer.decode(rest[j + 1:], skip_special=True)
        content = f"{before}<|think|>{think}<|/think|>{answer}"
        visible = f"{before}<|think|>{think}<|/think|>{answer}" if show_thinking else (before + answer)
        return content, visible.strip()
    think = tokenizer.decode(rest, skip_special=True)
    content = f"{before}<|think|>{think}"
    return content, (content if show_thinking else (before + think)).strip()


class _TextStreamer:
    """Token-id callback for ``generate(stream=...)`` that forwards decoded text deltas.

    Tool-call replies are muted entirely; thinking is muted unless ``show_thinking``; incomplete
    UTF-8 sequences are held back until complete.
    """

    def __init__(self, tokenizer: Any, callback: Callable[[str], Any], show_thinking: bool = False) -> None:
        self.tok = tokenizer
        self.callback = callback
        self.show_thinking = show_thinking
        self.ids: List[int] = []
        self.emitted = 0
        self.first = True
        self.muted = False
        self.in_think = False

    def __call__(self, token_id: int) -> None:
        tid = int(token_id)
        if self.first:
            self.first = False
            if tid == TOOL_CALL_ID:
                self.muted = True
        if self.muted:
            return
        if tid == THINK_ID:
            self.in_think = True
            return
        if tid == THINK_END_ID:
            self.in_think = False
            return
        if (self.in_think and not self.show_thinking) or tid < NUM_SPECIAL_TOKENS:
            return
        self.ids.append(tid)
        text = self.tok.decode(self.ids, skip_special=True)
        if text.endswith("�"):
            return
        delta = text[self.emitted:]
        if delta:
            self.emitted = len(text)
            self.callback(delta)


# -------------------------------------------------------------------------------------------------
# ChatSession
# -------------------------------------------------------------------------------------------------

class ChatSession:
    """Stateful multi-turn conversation with an :class:`~moonlight.llm.LLM` (SPEC §8).

    Args:
        llm: the ``LLM`` (needs ``.model``, ``.config`` and a tokenizer, see ``LLM.require_tokenizer``).
        system: system prompt; default ``llm.config.chat.system_prompt``.
        tools: None (no tools), True (built-in tools), a ``ToolRegistry``, or a list of ``Tool`` /
            functions / schema dicts / built-in names (see :func:`moonlight.tools.as_registry`).
        max_tool_rounds: tool calls allowed per :meth:`send` before an answer is forced.
        history: optional initial messages (copied). A leading system message becomes ``system``
            when ``system`` is not given.
        allow_code_exec: enables ``python_exec`` when ``tools=True`` / names it.
        **gen: generation defaults (keys in :data:`GEN_KEYS`), overriding ``config.chat``.

    Attributes:
        history: list of SPEC §5 messages (no system message).
        tool_log: ``[{"name", "arguments", "result"}]`` for every executed tool call.
        last_result: the last ``GenerationResult``.
    """

    def __init__(self, llm: Any, system: Optional[str] = None, tools: Any = None, max_tool_rounds: int = 4,
                 history: Optional[Sequence[Mapping[str, Any]]] = None, allow_code_exec: bool = False,
                 **gen: Any) -> None:
        self.llm = llm
        self._explicit_system = system is not None
        self.system: str = system if system is not None else str(llm.config.chat.system_prompt)
        self.registry: Optional[ToolRegistry] = as_registry(tools, allow_code_exec=allow_code_exec)
        self.max_tool_rounds = max(0, int(max_tool_rounds))
        resolve_generation(llm.config.chat, gen)  # validate keys early
        self.gen: Dict[str, Any] = dict(gen)
        self.tool_log: List[Dict[str, Any]] = []
        self.last_result: Any = None
        self._history: List[Dict[str, Any]] = []
        self._features: List[Any] = []  # per history message: image features (n_img, n_tok, vdim) or None
        if history:
            self.history = history  # type: ignore[assignment]

    # ------------------------------------------------------------------ history
    @property
    def history(self) -> List[Dict[str, Any]]:
        return self._history

    @history.setter
    def history(self, messages: Sequence[Mapping[str, Any]]) -> None:
        msgs = [dict(m) for m in (messages or [])]
        if msgs and str(msgs[0].get("role", "")).lower() == "system":
            first = msgs.pop(0)
            if not self._explicit_system:
                self.system = str(first.get("content") or "")
        self._history = msgs
        self._features = [None] * len(msgs)

    def reset(self) -> None:
        """Forget the conversation (system prompt and tools are kept)."""
        self._history = []
        self._features = []
        self.tool_log = []
        self.last_result = None

    def _append(self, message: Dict[str, Any], features: Any = None) -> None:
        self._history.append(message)
        self._features.append(features)

    @property
    def tools(self) -> Optional[ToolRegistry]:
        return self.registry

    def __repr__(self) -> str:
        tools = ", ".join(self.registry.names()) if self.registry else "none"
        return f"ChatSession(messages={len(self._history)}, tools=[{tools}])"

    # ------------------------------------------------------------------ prompt
    def _n_image_tokens(self) -> int:
        mm = self.llm.config.multimodal
        return int(mm.n_image_tokens) if bool(mm.enabled) else 0

    def _encode(self, msgs: Sequence[Mapping[str, Any]]) -> List[int]:
        tok = self.llm.require_tokenizer()
        schemas = self.registry.schemas() if (self.registry is not None) else None
        # The tools block stays in the prompt even when calls are no longer allowed so the
        # conversation keeps the format seen in training; the <|tool_call|> id is banned instead.
        ids, _ = tok.encode_chat(list(msgs), tools=schemas or None, add_generation_prompt=True,
                                 n_image_tokens=self._n_image_tokens(), system=self.system)
        return ids

    def _build_prompt(self, max_new_tokens: int) -> Tuple[List[int], Any, int]:
        """Render the history into prompt ids that fit ``model.max_seq_len``.

        Oldest turns are dropped first (the current user turn and the latest exchange are kept);
        as a last resort the prompt is left-truncated (not possible with images). Returns
        ``(ids, image_features or None, max_new_tokens)``.
        """
        max_len = int(self.llm.config.model.max_seq_len)
        reserve = min(int(max_new_tokens), max(1, max_len // 2))
        budget = max(8, max_len - reserve)
        msgs = list(self._history)
        feats = list(self._features)
        ids = self._encode(msgs)
        while len(ids) > budget:
            users = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
            last_user = users[-1] if users else 0
            if last_user > 0:
                del msgs[0], feats[0]
            elif len(msgs) - last_user - 1 > 2:
                del msgs[last_user + 1], feats[last_user + 1]
                while len(msgs) - last_user - 1 > 2 and msgs[last_user + 1].get("role") == "tool":
                    del msgs[last_user + 1], feats[last_user + 1]
            else:
                break
            ids = self._encode(msgs)

        for m, f in zip(msgs, feats):
            if m.get("role") == "user" and m.get("images") and f is None:
                raise ValueError("a history message declares images but no image features are available "
                                 "(pass images through ChatSession.send(..., images=...))")
        kept = [f for m, f in zip(msgs, feats) if f is not None and m.get("role") == "user" and m.get("images")]
        if len(ids) > budget:
            if kept:
                raise ValueError(f"prompt with images is longer than the model context ({len(ids)} > {budget} "
                                 f"tokens incl. room for the reply)")
            ids = [BOS_ID] + ids[-(budget - 1):]
        image_features = None
        if kept:
            import torch

            image_features = kept[0] if len(kept) == 1 else torch.cat(list(kept), dim=0)
        return ids, image_features, max(1, min(int(max_new_tokens), max_len - len(ids)))

    # ------------------------------------------------------------------ generation
    def _generate(self, g: Mapping[str, Any], allow_tools: bool) -> Any:
        tok = self.llm.require_tokenizer()
        ids, feats, max_new = self._build_prompt(int(g["max_new_tokens"]))
        banned = banned_special_ids(tok.vocab_size, int(self.llm.config.model.vocab_size), allow_tools)
        banned = sorted(set(banned) | {int(b) for b in (g.get("banned_ids") or ())})
        streamer = _TextStreamer(tok, g["stream"], bool(g["show_thinking"])) if g.get("stream") else None
        res = self.llm.generate_ids(
            ids, max_new_tokens=max_new, temperature=g["temperature"], top_k=g["top_k"], top_p=g["top_p"],
            repetition_penalty=g["repetition_penalty"], no_repeat_ngram=g["no_repeat_ngram"], n=g["n"],
            seed=g["seed"], stop_ids=(END_ID, EOS_ID), banned_ids=banned, image_features=feats,
            stream=streamer)
        self.last_result = res
        return res

    def send(self, message: str, images: Any = None, **gen: Any) -> str:
        """Add a user message, run the tool loop and return the assistant's reply text.

        Args:
            message: user text.
            images: one image or a list (PIL images, file paths or URLs); requires
                ``multimodal.enabled``. Encoded with ``LLM.encode_images``.
            **gen: per-call generation overrides (:data:`GEN_KEYS`).

        Returns:
            The final answer (thinking removed unless ``show_thinking``). The history receives the
            user message, every tool call / result, and the answer.
        """
        if not isinstance(message, str):
            raise TypeError(f"message must be a str, got {type(message).__name__}")
        g = resolve_generation(self.llm.config.chat, self.gen, gen)
        tok = self.llm.require_tokenizer()

        user: Dict[str, Any] = {"role": "user", "content": message}
        feats = None
        if images is not None:
            imgs = list(images) if isinstance(images, (list, tuple)) else [images]
            if imgs:
                if not bool(self.llm.config.multimodal.enabled):
                    raise ValueError(f"model {self.llm.config.name!r} is not multimodal "
                                     f"(multimodal.enabled is false); cannot take images")
                feats = self.llm.encode_images(imgs)
                user["images"] = len(imgs)
        self._append(user, feats)

        rounds = 0
        while True:
            allow_tools = self.registry is not None and rounds < self.max_tool_rounds
            res = self._generate(g, allow_tools)
            tokens = list(res.tokens)
            if allow_tools and tokens and tokens[0] == TOOL_CALL_ID:
                call = parse_tool_call(tok.decode(tokens[1:], skip_special=True))
                if call is not None:
                    rounds += 1
                    result = self.registry.call(call["name"], call["arguments"])  # never raises
                    self._append({"role": "assistant", "content": "", "tool_calls": [call]})
                    self._append({"role": "tool", "name": call["name"], "content": result})
                    self.tool_log.append({"name": call["name"], "arguments": call["arguments"], "result": result})
                    continue
                tokens = tokens[1:]  # unparsable call: keep the text as a plain answer
            content, visible = split_reply(tok, tokens, bool(g["show_thinking"]))
            self._append({"role": "assistant", "content": content})
            return visible

    __call__ = send

    def ask(self, message: str, **kw: Any) -> str:
        """Alias of :meth:`send`."""
        return self.send(message, **kw)


# -------------------------------------------------------------------------------------------------
# Agentic project building (§13.6)
# -------------------------------------------------------------------------------------------------

@dataclass
class ProjectResult:
    """Outcome of :func:`build_project` / ``LLM.build_project``.

    Attributes:
        workdir: project directory.
        files: files in the workspace afterwards (relative POSIX paths).
        tests_passed: result of a final ``run_tests`` (False if there are no tests or exec is off).
        steps: number of tool calls the agent executed.
        transcript: the conversation (SPEC §5 messages: the builder system prompt, then user /
            assistant tool calls / tool results / answers).
        final_answer: the agent's last reply.
        test_output: output of the final test run ("" if none).
    """

    workdir: str
    files: List[str]
    tests_passed: bool
    steps: int
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    test_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_project(llm: Any, task: str, workdir: Any, max_steps: int = 24, allow_exec: bool = True,
                  timeout: float = 20.0, max_nudges: int = 3, **gen: Any) -> ProjectResult:
    """Let the model build a small project in ``workdir`` with workspace tools.

    A :class:`ChatSession` with :data:`BUILDER_SYSTEM_PROMPT` and
    :func:`moonlight.tools.workspace_tools` receives ``task``; the tool loop runs for at most
    ``max_steps`` tool calls in total. When the model gives a final answer while the latest
    ``run_tests`` did not pass (or never ran), it is nudged with :data:`BUILD_NUDGE` (at most
    ``max_nudges`` times, while steps remain). It therefore stops early as soon as the tests pass
    and the model answers. Finally the tests are run once more to fill ``tests_passed``.
    ``max_new_tokens`` defaults to ``max(config.chat.max_new_tokens, 512)`` (files need room).
    """
    ws = Workspace(workdir, allow_exec=allow_exec, timeout=timeout)
    gen = dict(gen)
    if gen.get("max_new_tokens") is None:
        gen["max_new_tokens"] = max(int(llm.config.chat.max_new_tokens), 512)
    max_steps = max(1, int(max_steps))
    session = ChatSession(llm, system=BUILDER_SYSTEM_PROMPT, tools=workspace_tools(ws), max_tool_rounds=max_steps,
                          **gen)
    final = session.send(task)
    nudges = 0
    while allow_exec and nudges < max(0, int(max_nudges)):
        remaining = max_steps - len(session.tool_log)
        if remaining <= 0 or (ws.last_test is not None and ws.last_test.get("passed")):
            break
        session.max_tool_rounds = remaining
        final = session.send(BUILD_NUDGE)
        nudges += 1

    tests_passed, test_output = False, ""
    if allow_exec and ws.test_files():
        result = ws.run_tests()
        tests_passed, test_output = bool(result["passed"]), str(result["output"])
    return ProjectResult(workdir=str(ws.root), files=ws.files(), tests_passed=tests_passed,
                         steps=len(session.tool_log),
                         transcript=[{"role": "system", "content": session.system}]
                         + copy.deepcopy(session.history),
                         final_answer=final, test_output=test_output)
