"""Moonlight tokenizer: byte-level BPE, fixed special tokens and the chat template (SPEC §5).

Overview
--------
* Backend: Hugging Face ``tokenizers``. It is imported lazily (inside functions), so the constants
  of this module (``END_ID``, ``SPECIAL_TOKENS``, ``DEFAULT_SYSTEM_PROMPT`` ...) can be imported
  without it; torch is never needed.
* Special tokens have FIXED ids ``0..31`` (``SPECIAL_TOKENS``). An untrained
  :meth:`MoonlightTokenizer.byte_level` tokenizer has exactly 288 tokens: ids ``0..31`` are the
  specials and id ``32 + b`` is the raw byte ``b`` (``b`` in ``0..255``).
* Pipeline: apostrophe normalization (:func:`normalize_text`) -> ``Split(PRETOKENIZE_PATTERN,
  "isolated")`` -> ``ByteLevel(add_prefix_space=False, use_regex=False)`` -> BPE; decoder
  ``ByteLevel``. Because every byte has a token, ``decode(encode(text)) == normalize_text(text)``
  for ANY Unicode string (Uzbek, English, code with tabs/newlines, emoji, digits, control chars).
* Chat template (no newlines between parts)::

      <|bos|><|system|>{system}[<|tools|>{tools json}]<|end|>
      <|user|>[<|image|> x n_image_tokens per image]{content}<|end|>
      <|assistant|>{content}<|end|>
      <|assistant|><|tool_call|>{"name": "...", "arguments": {...}}<|end|>
      <|tool_result|>{content}<|end|>

  ``loss_mask`` is 1 on assistant content, on ``<|tool_call|>`` + its JSON and on the closing
  ``<|end|>`` of assistant messages; 0 everywhere else (``<|assistant|>`` headers included).

Normalization note
------------------
SPEC §5 maps the apostrophe look-alikes ``ʻ ʼ ‘ ’ ` ´`` to ``'``. The grave accent (backtick) is
mapped ONLY when it sits between two letters (Uzbek typing such as ``o`zbek``/``g`alaba``), so
Markdown code fences, inline code, shell command substitution and JS template strings survive
untouched -- mapping every backtick would corrupt code, which Moonlight models are trained to
write. All other look-alikes are always mapped.

Special tokens inside text
--------------------------
:meth:`MoonlightTokenizer.encode` / :meth:`~MoonlightTokenizer.encode_batch` recognize
special-token strings by default (``encode("<|end|>") == [END_ID]``), like Hugging Face
tokenizers. The chat template is injection-safe: system / user / tool text and the tool JSON are
encoded literally (a user typing ``<|end|>`` or ``<|image|>`` gets plain byte tokens), while
assistant content recognizes only ``ASSISTANT_INLINE_SPECIAL_IDS`` (``<|think|>``,
``<|/think|>``, ``<|tool_call|>``, ``<|code|>``) -- reasoning data written as
``"<|think|>...<|/think|>answer"`` therefore gets the real control tokens.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Collection, Dict, FrozenSet, Iterable, Iterator, List,
                    Mapping, NamedTuple, Optional, Sequence, Tuple, Union)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tokenizers

__all__ = [
    # special tokens / ids
    "SPECIAL_TOKENS", "SPECIAL_TOKEN_IDS", "NUM_SPECIAL_TOKENS", "NUM_BYTE_TOKENS", "BYTE_OFFSET",
    "MIN_VOCAB_SIZE", "PAD_ID", "BOS_ID", "EOS_ID", "SYSTEM_ID", "USER_ID", "ASSISTANT_ID", "END_ID",
    "IMAGE_ID", "TOOL_CALL_ID", "TOOL_RESULT_ID", "TOOLS_ID", "THINK_ID", "THINK_END_ID", "CODE_ID",
    "AUDIO_ID", "STOP_IDS", "ASSISTANT_INLINE_SPECIAL_IDS", "DEFAULT_SYSTEM_PROMPT",
    # text processing
    "PRETOKENIZE_PATTERN", "PRETOKENIZE_PATTERN_GROUPED_DIGITS", "APOSTROPHE_VARIANTS",
    "pretokenize_pattern", "normalize_text", "bytes_to_unicode", "is_special",
    "format_tools", "format_tool_call",
    # tokenizer
    "MoonlightTokenizer", "train_tokenizer",
]

PathLike = Union[str, "os.PathLike[str]"]

# =================================================================================================
# Special tokens (ids FIXED -- never reorder)
# =================================================================================================

SPECIAL_TOKENS: List[str] = [
    "<|pad|>",          # 0
    "<|bos|>",          # 1
    "<|eos|>",          # 2
    "<|system|>",       # 3
    "<|user|>",         # 4
    "<|assistant|>",    # 5
    "<|end|>",          # 6
    "<|image|>",        # 7
    "<|tool_call|>",    # 8
    "<|tool_result|>",  # 9
    "<|tools|>",        # 10
    "<|think|>",        # 11
    "<|/think|>",       # 12
    "<|code|>",         # 13
    "<|audio|>",        # 14
] + [f"<|reserved_{i}|>" for i in range(17)]  # 15..31

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SYSTEM_ID = 3
USER_ID = 4
ASSISTANT_ID = 5
END_ID = 6
IMAGE_ID = 7
TOOL_CALL_ID = 8
TOOL_RESULT_ID = 9
TOOLS_ID = 10
THINK_ID = 11
THINK_END_ID = 12
CODE_ID = 13
AUDIO_ID = 14

NUM_SPECIAL_TOKENS = 32
NUM_BYTE_TOKENS = 256
#: id of byte ``b`` in the untrained :meth:`MoonlightTokenizer.byte_level` tokenizer is ``BYTE_OFFSET + b``.
BYTE_OFFSET = NUM_SPECIAL_TOKENS
#: smallest valid vocabulary: 32 specials + 256 byte tokens.
MIN_VOCAB_SIZE = NUM_SPECIAL_TOKENS + NUM_BYTE_TOKENS

#: token string -> fixed id, for all 32 special tokens.
SPECIAL_TOKEN_IDS: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}

#: generation stops on either of these (same order as ``generation.generate(stop_ids=...)``).
STOP_IDS: Tuple[int, ...] = (END_ID, EOS_ID)

#: special tokens recognized inside ASSISTANT message content by the chat template; every other
#: special-token string in chat content is encoded literally (see module docstring).
ASSISTANT_INLINE_SPECIAL_IDS: FrozenSet[int] = frozenset({TOOL_CALL_ID, THINK_ID, THINK_END_ID, CODE_ID})

DEFAULT_SYSTEM_PROMPT = (
    "Sen Moonlight — o'zbek va ingliz tillarida samimiy, aniq va foydali javob beradigan, kod yoza "
    "oladigan va kerak bo'lganda vositalardan (tools) foydalanadigan sun'iy intellekt yordamchisisan."
)

_NO_SPECIALS: FrozenSet[int] = frozenset()
_ALL_SPECIALS: FrozenSet[int] = frozenset(range(NUM_SPECIAL_TOKENS))
# No special token is a prefix of another (all end with "|>"), longest-first only for clarity.
_SPECIAL_RE = re.compile("|".join(re.escape(t) for t in sorted(SPECIAL_TOKENS, key=len, reverse=True)))

# =================================================================================================
# Pre-tokenization regex and normalization
# =================================================================================================

#: Split regex (Oniguruma syntax): words with Uzbek/English apostrophes kept whole (o'zbek, don't),
#: single digits, punctuation runs, whitespace (GPT-2 style ``\s+(?!\S)`` keeps the last space
#: attached to the next word).
PRETOKENIZE_PATTERN = r"[ ]?\p{L}+(?:['ʻʼ‘’]\p{L}+)*| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
#: Variant used when ``split_digits=False``: the digit atom ``\p{N}`` becomes ``\p{N}{1,3}``.
PRETOKENIZE_PATTERN_GROUPED_DIGITS = (
    r"[ ]?\p{L}+(?:['ʻʼ‘’]\p{L}+)*| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

#: Apostrophe look-alikes normalized to ``'`` (SPEC §5): U+02BB ʻ, U+02BC ʼ, U+2018 ‘, U+2019 ’,
#: U+0060 ` (only between two letters, see module docstring) and U+00B4 ´.
APOSTROPHE_VARIANTS = "\u02bb\u02bc\u2018\u2019\u0060\u00b4"
_ALWAYS_MAPPED = "\u02bb\u02bc\u2018\u2019\u00b4"
_BACKTICK_RE = re.compile(r"(?<=\w)`(?=\w)")
_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def pretokenize_pattern(split_digits: bool = True) -> str:
    """Return the Split pre-tokenizer regex (single digits if ``split_digits`` else 1-3 digits)."""
    return PRETOKENIZE_PATTERN if split_digits else PRETOKENIZE_PATTERN_GROUPED_DIGITS


def _backtick_repl(match: "re.Match[str]") -> str:
    s, i = match.string, match.start()
    # ``\w`` also matches digits/underscore; ``str.isalpha`` is exactly Unicode ``\p{L}``.
    return "'" if s[i - 1].isalpha() and s[i + 1].isalpha() else "`"


def normalize_text(text: str) -> str:
    """Map apostrophe look-alikes to ``'`` (SPEC §5).

    ``ʻ ʼ ‘ ’ ´`` are always mapped; the backtick only between two letters (``o`zbek`` ->
    ``o'zbek``, but ```` ```python ```` and ```` `x` ```` are kept). Idempotent; nothing else
    (whitespace, case, Unicode form) is changed, so the tokenizer round-trips everything else.
    """
    for ch in _ALWAYS_MAPPED:
        if ch in text:
            text = text.replace(ch, "'")
    if "`" in text:
        text = _BACKTICK_RE.sub(_backtick_repl, text)
    return text


def _replace_surrogates(text: str) -> str:
    """Lone UTF-16 surrogates (e.g. from ``errors='surrogateescape'``) cannot be encoded to UTF-8;
    replace each with U+FFFD so they never crash tokenization."""
    return _SURROGATE_RE.sub("\ufffd", text)


def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 byte -> printable-character table used by the ``ByteLevel`` pre-tokenizer.

    Returns a dict with 256 entries ``{byte: char}``; its values equal
    ``tokenizers.pre_tokenizers.ByteLevel.alphabet()`` (as a set).
    """
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def is_special(token_id: Any) -> bool:
    """True iff ``token_id`` is one of the 32 fixed special-token ids (``0 <= id < 32``)."""
    try:
        i = int(token_id)
    except (TypeError, ValueError):
        return False
    return 0 <= i < NUM_SPECIAL_TOKENS


# =================================================================================================
# Tool rendering helpers
# =================================================================================================

def format_tools(schemas: Sequence[Any]) -> str:
    """Serialize tool schemas for the ``<|tools|>`` block (compact JSON, non-ASCII kept)."""
    return json.dumps(list(schemas), ensure_ascii=False, separators=(",", ":"), default=str)


def format_tool_call(name: str, arguments: Any = None) -> str:
    """Serialize one tool call: ``{"name": "...", "arguments": {...}}`` (default separators,
    non-ASCII kept; ``arguments=None`` -> ``{}``)."""
    return json.dumps({"name": name, "arguments": {} if arguments is None else arguments},
                      ensure_ascii=False, default=str)


def _tool_schemas(tools: Any) -> List[Any]:
    """Normalize ``tools`` to a list of JSON-able schemas.

    Accepts None, a list of dict schemas (used verbatim, any style), a single dict, objects exposing
    ``schema``/``schema()``/``to_schema()``, objects with ``name``/``description``/``parameters``
    attributes (e.g. ``moonlight.tools.Tool``) or a registry exposing ``schemas()``.
    """
    if tools is None:
        return []
    schemas_fn = getattr(tools, "schemas", None)
    if callable(schemas_fn):
        tools = schemas_fn()
    if isinstance(tools, Mapping):
        tools = [tools]
    out: List[Any] = []
    for i, tool in enumerate(tools):
        if isinstance(tool, Mapping):
            out.append(tool)
            continue
        schema = getattr(tool, "schema", None)
        if schema is None:
            schema = getattr(tool, "to_schema", None)
        if callable(schema):
            schema = schema()
        if isinstance(schema, Mapping):
            out.append(schema)
        elif hasattr(tool, "name") and hasattr(tool, "parameters"):
            out.append({"name": tool.name, "description": getattr(tool, "description", "") or "",
                        "parameters": tool.parameters})
        else:
            raise TypeError(f"tools[{i}]: cannot build a JSON schema from {type(tool).__name__}")
    return out


def _tool_call_json(call: Any, where: str) -> str:
    """Render one entry of ``message["tool_calls"]``.

    Accepts ``{"name", "arguments"}`` (our format) or OpenAI style ``{"function": {"name",
    "arguments"}}``; ``arguments`` given as a JSON string is parsed when possible.
    """
    if isinstance(call, str):
        try:
            call = json.loads(call)
        except ValueError:
            raise ValueError(f"{where}: tool call string is not valid JSON") from None
    if not isinstance(call, Mapping):
        raise TypeError(f"{where}: tool call must be a dict, got {type(call).__name__}")
    fn = call.get("function")
    src = fn if isinstance(fn, Mapping) else call
    name = src.get("name")
    if not name:
        raise ValueError(f"{where}: tool call has no 'name'")
    args = src.get("arguments", {})
    if isinstance(args, str):
        stripped = args.strip()
        if not stripped:
            args = {}
        else:
            try:
                args = json.loads(stripped)
            except ValueError:
                pass  # keep the raw string; still valid JSON once dumped
    return format_tool_call(str(name), args)


def _content_text(content: Any) -> str:
    """Message content -> str (None -> "", dict/list -> compact-ish JSON, other -> str())."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (Mapping, list, tuple)):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)


def _image_count(value: Any, where: str) -> int:
    """``message["images"]``: an int count (SPEC) or a list of images (its length)."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise TypeError(f"{where}: 'images' must be an int count, got {type(value).__name__}") from None
    if n < 0:
        raise ValueError(f"{where}: 'images' must be >= 0, got {n}")
    return n


# =================================================================================================
# Chat template -> segments (pure python; shared by encode_chat and render_chat)
# =================================================================================================

class _Segment(NamedTuple):
    """One piece of a rendered conversation.

    ``token`` >= 0: a run of ``repeat`` copies of special token ``token`` (``text`` unused);
    ``token`` == -1: ``text`` to tokenize, recognizing only special-token strings in ``allowed``.
    ``train``: loss-mask value for every token of the piece.
    """

    token: int
    text: str
    repeat: int
    train: bool
    allowed: FrozenSet[int]


def _tok(token_id: int, train: bool = False, repeat: int = 1) -> _Segment:
    return _Segment(token_id, "", repeat, train, _NO_SPECIALS)


def _txt(text: str, train: bool = False, allowed: FrozenSet[int] = _NO_SPECIALS) -> _Segment:
    return _Segment(-1, text, 1, train, allowed)


_ROLES = ("system", "user", "assistant", "tool")


def _chat_segments(messages: Sequence[Mapping[str, Any]], tools: Any, add_generation_prompt: bool,
                   n_image_tokens: int, system: Optional[str]) -> List[_Segment]:
    """Lay out the SPEC §5 chat template as segments (see :meth:`MoonlightTokenizer.encode_chat`)."""
    if isinstance(messages, Mapping):
        raise TypeError("messages must be a list of message dicts, not a single dict")
    msgs = list(messages)
    n_image_tokens = int(n_image_tokens)
    if n_image_tokens < 0:
        raise ValueError(f"n_image_tokens must be >= 0, got {n_image_tokens}")
    roles: List[str] = []
    for i, m in enumerate(msgs):
        if not isinstance(m, Mapping):
            raise TypeError(f"messages[{i}] must be a dict, got {type(m).__name__}")
        role = str(m.get("role", "")).strip().lower()
        if role not in _ROLES:
            raise ValueError(f"messages[{i}]: unknown role {m.get('role')!r}; expected one of {_ROLES}")
        roles.append(role)

    start = 0
    system_text = None if system is None else _content_text(system)
    if msgs and roles[0] == "system":
        start = 1  # consumed; an explicit ``system`` argument takes precedence over it
        if system_text is None:
            system_text = _content_text(msgs[0].get("content"))
    if system_text is None:
        system_text = DEFAULT_SYSTEM_PROMPT

    segs: List[_Segment] = [_tok(BOS_ID), _tok(SYSTEM_ID), _txt(system_text)]
    schemas = _tool_schemas(tools)
    if schemas:
        segs += [_tok(TOOLS_ID), _txt(format_tools(schemas))]
    segs.append(_tok(END_ID))

    for i in range(start, len(msgs)):
        m, role, where = msgs[i], roles[i], f"messages[{i}]"
        if role == "user":
            segs.append(_tok(USER_ID))
            n_img = _image_count(m.get("images"), where) * n_image_tokens
            if n_img:
                segs.append(_tok(IMAGE_ID, repeat=n_img))
            segs += [_txt(_content_text(m.get("content"))), _tok(END_ID)]
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            if isinstance(calls, Mapping):
                calls = [calls]
            if calls:
                # One tool call per assistant turn (SPEC §5; ChatSession detects a turn starting with
                # <|tool_call|>). Several calls become consecutive turns; free-text content of a
                # tool-calling message is not part of the template and is not rendered.
                for j, call in enumerate(calls):
                    segs += [_tok(ASSISTANT_ID), _tok(TOOL_CALL_ID, train=True),
                             _txt(_tool_call_json(call, f"{where}.tool_calls[{j}]"), train=True),
                             _tok(END_ID, train=True)]
            else:
                segs += [_tok(ASSISTANT_ID),
                         _txt(_content_text(m.get("content")), train=True,
                              allowed=ASSISTANT_INLINE_SPECIAL_IDS),
                         _tok(END_ID, train=True)]
        elif role == "tool":
            segs += [_tok(TOOL_RESULT_ID), _txt(_content_text(m.get("content"))), _tok(END_ID)]
        else:  # a system message after the first position
            segs += [_tok(SYSTEM_ID), _txt(_content_text(m.get("content"))), _tok(END_ID)]

    if add_generation_prompt:
        segs.append(_tok(ASSISTANT_ID))
    return segs


# =================================================================================================
# Backend helpers (lazy ``tokenizers`` import)
# =================================================================================================

def _tokenizers() -> Any:
    try:
        import tokenizers
    except ImportError as exc:  # pragma: no cover - hard dependency of the package
        raise ImportError("moonlight.tokenizer needs the 'tokenizers' package "
                          "(pip install 'tokenizers>=0.15')") from exc
    return tokenizers


def _special_added_tokens() -> List[Any]:
    tk = _tokenizers()
    return [tk.AddedToken(t, single_word=False, lstrip=False, rstrip=False, normalized=False,
                          special=True) for t in SPECIAL_TOKENS]


def _configure_backend(backend: Any, split_digits: bool) -> None:
    """Install Moonlight's normalizer, pre-tokenizer and decoder on a ``tokenizers.Tokenizer``.

    The normalizer mirrors :func:`normalize_text` so a saved ``tokenizer.json`` is self-contained
    (the Python wrapper also normalizes, which is idempotent).
    """
    tk = _tokenizers()
    backend.normalizer = tk.normalizers.Sequence([
        tk.normalizers.Replace(tk.Regex("[" + _ALWAYS_MAPPED + "]"), "'"),
        tk.normalizers.Replace(tk.Regex(r"(?<=\p{L})`(?=\p{L})"), "'"),
    ])
    backend.pre_tokenizer = tk.pre_tokenizers.Sequence([
        tk.pre_tokenizers.Split(tk.Regex(pretokenize_pattern(split_digits)), behavior="isolated"),
        tk.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    backend.decoder = tk.decoders.ByteLevel()


# =================================================================================================
# MoonlightTokenizer
# =================================================================================================

class MoonlightTokenizer:
    """Byte-level BPE tokenizer with Moonlight's fixed special tokens and chat template.

    Wraps a ``tokenizers.Tokenizer`` (``.backend``). Construct with :meth:`byte_level`,
    :func:`train_tokenizer`, :meth:`from_file` or :meth:`from_str`. All ids are python ``int`` and
    all sequences python lists. Instances are picklable (serialized as tokenizer JSON).
    """

    def __init__(self, backend: "tokenizers.Tokenizer") -> None:
        tk = _tokenizers()
        if not isinstance(backend, tk.Tokenizer):
            raise TypeError(f"expected tokenizers.Tokenizer, got {type(backend).__name__}")
        for i, tok in enumerate(SPECIAL_TOKENS):
            got = backend.token_to_id(tok)
            if got != i:
                raise ValueError(f"not a Moonlight tokenizer: special token {tok} has id {got}, "
                                 f"expected {i}")
        # Idempotent: registers the specials as *special added tokens* (atomic when encoding,
        # skippable when decoding) if a tokenizer.json only has them in the model vocabulary.
        backend.add_special_tokens(_special_added_tokens())
        vocab = backend.get_vocab(with_added_tokens=False)
        missing = [ch for ch in bytes_to_unicode().values() if ch not in vocab]
        if missing:
            raise ValueError(f"not a byte-level tokenizer: {len(missing)} of 256 byte tokens missing "
                             "(text could not round-trip)")
        self._backend = backend
        self._literal: Any = None  # lazily built clone that encodes special strings as plain text

    # ------------------------------------------------------------------------------ construction
    @classmethod
    def byte_level(cls) -> "MoonlightTokenizer":
        """Untrained tokenizer (tests / offline): ids 0..31 specials, id ``32 + b`` = byte ``b``.

        ``vocab_size == 288``; every UTF-8 byte is one token (no merges).
        """
        tk = _tokenizers()
        vocab: Dict[str, int] = dict(SPECIAL_TOKEN_IDS)
        for b, ch in bytes_to_unicode().items():
            vocab[ch] = BYTE_OFFSET + b
        backend = tk.Tokenizer(tk.models.BPE(vocab=vocab, merges=[]))
        _configure_backend(backend, split_digits=True)
        backend.add_special_tokens(_special_added_tokens())
        return cls(backend)

    @classmethod
    def from_file(cls, path: PathLike) -> "MoonlightTokenizer":
        """Load a ``tokenizer.json`` (``path`` may also be a directory containing one)."""
        p = Path(path).expanduser()
        if p.is_dir():
            p = p / "tokenizer.json"
        if not p.is_file():
            raise FileNotFoundError(f"tokenizer file not found: {p}")
        return cls(_tokenizers().Tokenizer.from_file(str(p)))

    @classmethod
    def from_str(cls, s: str) -> "MoonlightTokenizer":
        """Build from a tokenizer JSON string (as produced by :meth:`to_str`)."""
        return cls(_tokenizers().Tokenizer.from_str(s))

    def to_str(self) -> str:
        """Compact tokenizer JSON (stored in checkpoints / exported models)."""
        return self._backend.to_str()

    def save(self, path: PathLike) -> Path:
        """Atomically write pretty tokenizer JSON to ``path`` (a directory -> ``path/tokenizer.json``).

        Parent directories are created. Returns the file path written.
        """
        p = Path(path).expanduser()
        if p.is_dir():
            p = p / "tokenizer.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self._backend.to_str(pretty=True)
        fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o644)  # mkstemp creates 0600; a tokenizer file is not secret
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return p

    def __getstate__(self) -> Dict[str, Any]:
        return {"tokenizer_json": self.to_str()}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__init__(_tokenizers().Tokenizer.from_str(state["tokenizer_json"]))  # type: ignore[misc]

    # ------------------------------------------------------------------------------ properties
    @property
    def backend(self) -> "tokenizers.Tokenizer":
        """The wrapped ``tokenizers.Tokenizer``."""
        return self._backend

    @property
    def vocab_size(self) -> int:
        """Number of token ids (specials included); every id is in ``[0, vocab_size)``.

        A trained tokenizer may be smaller than the requested size when the corpus runs out of
        merges; it is never larger.
        """
        return int(self._backend.get_vocab_size(with_added_tokens=True))

    @property
    def special_ids(self) -> Dict[str, int]:
        """``{"<|pad|>": 0, "<|bos|>": 1, ..., "<|reserved_16|>": 31}`` (a fresh copy)."""
        return dict(SPECIAL_TOKEN_IDS)

    @property
    def stop_ids(self) -> Tuple[int, ...]:
        """Ids that end generation: ``(END_ID, EOS_ID)``."""
        return STOP_IDS

    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID
    end_id = END_ID
    image_id = IMAGE_ID

    @staticmethod
    def is_special(token_id: Any) -> bool:
        """True iff ``token_id`` is one of the 32 special-token ids."""
        return is_special(token_id)

    def token_to_id(self, token: str) -> Optional[int]:
        """Vocabulary lookup of a token string (byte-level form for normal tokens)."""
        return self._backend.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Token string of an id (byte-level form for normal tokens), None if out of range."""
        return self._backend.id_to_token(int(token_id))

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"MoonlightTokenizer(vocab_size={self.vocab_size})"

    # ------------------------------------------------------------------------------ encoding
    @staticmethod
    def _prepare(text: Any) -> str:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        return normalize_text(text)

    def _literal_backend(self) -> Any:
        """Clone of the backend with ``encode_special_tokens=True`` (special-token strings are
        encoded as ordinary text), or False if this ``tokenizers`` version lacks the switch."""
        if self._literal is None:
            clone = _tokenizers().Tokenizer.from_str(self._backend.to_str())
            if hasattr(clone, "encode_special_tokens"):
                clone.encode_special_tokens = True
                self._literal = clone
            else:  # pragma: no cover - tokenizers < 0.15.1
                self._literal = False
        return self._literal

    def _backend_encode(self, texts: Sequence[str], literal: bool) -> List[List[int]]:
        """Encode already-normalized texts with one backend call.

        ``literal=False``: special-token strings become their ids; ``literal=True``: they are plain
        text. Returns one id list per text (``[]`` for empty strings).
        """
        out: List[List[int]] = [[] for _ in texts]
        idx = [i for i, t in enumerate(texts) if t]
        if not idx:
            return out
        backend = self._literal_backend() if literal else self._backend
        if backend is False:  # pragma: no cover - old tokenizers: break special strings apart
            return self._encode_literal_fallback(texts)
        todo = [texts[i] for i in idx]
        try:
            ids = self._run_backend(backend, todo)
        except (TypeError, ValueError, UnicodeError):
            ids = self._run_backend(backend, [_replace_surrogates(t) for t in todo])
        for i, row in zip(idx, ids):
            out[i] = row
        return out

    @staticmethod
    def _run_backend(backend: Any, texts: List[str]) -> List[List[int]]:
        if len(texts) == 1:
            return [backend.encode(texts[0], add_special_tokens=False).ids]
        fast = getattr(backend, "encode_batch_fast", None)  # tokenizers >= 0.20: no offsets
        encs = (fast or backend.encode_batch)(texts, add_special_tokens=False)
        return [e.ids for e in encs]

    def _encode_literal_fallback(self, texts: Sequence[str]) -> List[List[int]]:  # pragma: no cover
        """Literal encoding without ``encode_special_tokens``: split every special-token string
        after its first character so the backend cannot match it (decodes to the same text)."""
        pieces: List[List[str]] = []
        for t in texts:
            parts: List[str] = []
            last = 0
            for m in _SPECIAL_RE.finditer(t):
                parts += [t[last:m.start() + 1]]
                last = m.start() + 1
            parts.append(t[last:])
            pieces.append([p for p in parts if p])
        flat = [p for ps in pieces for p in ps]
        enc = self._backend_encode(flat, literal=False) if flat else []
        out: List[List[int]] = []
        k = 0
        for ps in pieces:
            row: List[int] = []
            for _ in ps:
                row.extend(enc[k])
                k += 1
            out.append(row)
        return out

    def _encode_split(self, texts: Sequence[str], allowed: Sequence[FrozenSet[int]]) -> List[List[int]]:
        """Encode normalized ``texts``; in text ``i`` only special-token strings whose id is in
        ``allowed[i]`` become control tokens, everything else is literal. One backend call."""
        plans: List[List[Tuple[bool, int]]] = []  # (is_special, id_or_piece_index)
        pieces: List[str] = []
        for text, allow in zip(texts, allowed):
            plan: List[Tuple[bool, int]] = []
            last = 0
            if allow and "<|" in text:
                for m in _SPECIAL_RE.finditer(text):
                    sid = SPECIAL_TOKEN_IDS[m.group(0)]
                    if sid not in allow:
                        continue
                    if m.start() > last:
                        plan.append((False, len(pieces)))
                        pieces.append(text[last:m.start()])
                    plan.append((True, sid))
                    last = m.end()
            if last < len(text):
                plan.append((False, len(pieces)))
                pieces.append(text[last:])
            plans.append(plan)
        enc = self._backend_encode(pieces, literal=True)
        out: List[List[int]] = []
        for plan in plans:
            row: List[int] = []
            for is_sp, v in plan:
                if is_sp:
                    row.append(v)
                else:
                    row.extend(enc[v])
            out.append(row)
        return out

    def _encode_normalized(self, texts: Sequence[str], allow_special: Union[bool, Collection[int]]
                           ) -> List[List[int]]:
        if allow_special is True:
            return self._backend_encode(texts, literal=False)
        if allow_special is False:
            return self._backend_encode(texts, literal=True)
        allowed = frozenset(int(i) for i in allow_special)
        if allowed >= _ALL_SPECIALS:
            return self._backend_encode(texts, literal=False)
        return self._encode_split(texts, [allowed] * len(texts))

    def encode(self, text: str, add_bos: bool = False,
               allow_special: Union[bool, Collection[int]] = True) -> List[int]:
        """Text -> token ids ``list[int]`` (length >= 0; ``[BOS_ID] + ...`` if ``add_bos``).

        ``allow_special``: True (default) -> special-token strings in ``text`` (e.g. ``"<|end|>"``)
        become their ids; False -> encoded as plain text; a collection of ids -> only those.
        """
        ids = self._encode_normalized([self._prepare(text)], allow_special)[0]
        return [BOS_ID] + ids if add_bos else ids

    def encode_batch(self, texts: Iterable[str], add_bos: bool = False,
                     allow_special: Union[bool, Collection[int]] = True) -> List[List[int]]:
        """Encode many texts at once (parallel in the Rust backend); same semantics as :meth:`encode`.

        Returns ``len(texts)`` id lists.
        """
        if isinstance(texts, str):
            raise TypeError("encode_batch expects an iterable of strings, not a single str")
        rows = self._encode_normalized([self._prepare(t) for t in texts], allow_special)
        return [[BOS_ID] + r for r in rows] if add_bos else rows

    # ------------------------------------------------------------------------------ decoding
    def _clean_ids(self, ids: Any) -> List[int]:
        if hasattr(ids, "tolist"):  # numpy array / torch tensor
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        n = self.vocab_size
        out: List[int] = []
        for i in ids:
            i = int(i)
            if 0 <= i < n:  # ids beyond the tokenizer (e.g. sampled by a random model) are dropped
                out.append(i)
        return out

    def decode(self, ids: Any, skip_special: bool = True) -> str:
        """Token ids (list / tuple / 1-D numpy array / tensor) -> text.

        ``skip_special`` drops the 32 special tokens; otherwise they appear as their strings
        (``"<|end|>"``). Ids outside ``[0, vocab_size)`` are ignored; byte sequences that are not
        valid UTF-8 (e.g. a truncated multi-byte char) decode to U+FFFD.
        """
        return self._backend.decode(self._clean_ids(ids), skip_special_tokens=skip_special)

    def decode_batch(self, batch: Iterable[Any], skip_special: bool = True) -> List[str]:
        """Decode several id sequences (see :meth:`decode`)."""
        return [self.decode(ids, skip_special=skip_special) for ids in batch]

    # ------------------------------------------------------------------------------ chat
    def encode_chat(self, messages: Sequence[Mapping[str, Any]], tools: Any = None,
                    add_generation_prompt: bool = False, n_image_tokens: int = 0,
                    system: Optional[str] = None) -> Tuple[List[int], List[int]]:
        """Render a conversation with the SPEC §5 template and tokenize it.

        Args:
            messages: list of ``{"role": "system"|"user"|"assistant"|"tool", "content": str}``;
                assistant messages may carry ``"tool_calls": [{"name", "arguments"}]`` (rendered one
                ``<|assistant|><|tool_call|>{json}<|end|>`` turn per call; free-text content of a
                tool-calling message is not rendered); user messages may carry ``"images": int``
                (``images * n_image_tokens`` ``<|image|>`` tokens precede the content); tool
                messages may carry ``"name"`` (not rendered). A leading system message is the system
                prompt; later system messages render as ``<|system|>{content}<|end|>``.
            tools: tool schemas (list of dicts, ``Tool`` objects or a registry) -> ``<|tools|>`` +
                compact JSON inside the system block; None / empty -> no tools block.
            add_generation_prompt: append ``<|assistant|>`` (mask 0) to prompt a reply.
            n_image_tokens: placeholders per image (``multimodal.n_image_tokens``; 0 -> none).
            system: system prompt; overrides a leading system message. If None and there is no
                leading system message, ``DEFAULT_SYSTEM_PROMPT`` is used.

        Returns:
            ``(ids, loss_mask)``: two python ``list[int]`` of equal length ``T``. ``loss_mask[t]``
            is 1 iff token ``t`` is a training target: assistant content tokens, ``<|tool_call|>``
            + JSON and the closing ``<|end|>`` of assistant turns.
        """
        segs = _chat_segments(messages, tools, add_generation_prompt, n_image_tokens, system)
        texts = [s for s in segs if s.token < 0 and s.text]
        encoded = self._encode_split([normalize_text(s.text) for s in texts], [s.allowed for s in texts])
        ids: List[int] = []
        mask: List[int] = []
        k = 0
        for s in segs:
            m = 1 if s.train else 0
            if s.token >= 0:
                ids.extend([s.token] * s.repeat)
                mask.extend([m] * s.repeat)
            elif s.text:
                row = encoded[k]
                k += 1
                ids.extend(row)
                mask.extend([m] * len(row))
        return ids, mask

    def render_chat(self, messages: Sequence[Mapping[str, Any]], tools: Any = None,
                    add_generation_prompt: bool = False, n_image_tokens: int = 0,
                    system: Optional[str] = None, *, collapse_images: bool = False,
                    mark_loss: bool = False) -> str:
        """Human-readable rendering of :meth:`encode_chat` (same arguments) for debugging.

        By default the result equals ``decode(encode_chat(...)[0], skip_special=False)``.
        ``collapse_images`` shows image placeholders as ``<|image|>×N``; ``mark_loss`` wraps every
        span with loss_mask 1 in ``⟦ ⟧``.
        """
        segs = _chat_segments(messages, tools, add_generation_prompt, n_image_tokens, system)
        parts: List[str] = []
        in_loss = False
        for s in segs:
            if mark_loss and s.train != in_loss:
                parts.append("⟦" if s.train else "⟧")
                in_loss = s.train
            if s.token >= 0:
                tok = SPECIAL_TOKENS[s.token]
                if collapse_images and s.token == IMAGE_ID and s.repeat > 1:
                    parts.append(f"{tok}×{s.repeat}")
                else:
                    parts.append(tok * s.repeat)
            else:
                parts.append(_replace_surrogates(normalize_text(s.text)))
        if mark_loss and in_loss:
            parts.append("⟧")
        return "".join(parts)


# =================================================================================================
# Training
# =================================================================================================

def _training_stream(texts: Iterable[str], max_chars: Optional[int]) -> Iterator[str]:
    if isinstance(texts, str):
        texts = (texts,)
    budget = None if max_chars is None else int(max_chars)
    for text in texts:
        if budget is not None and budget <= 0:
            break
        if not isinstance(text, str):
            raise TypeError(f"train_tokenizer: texts must be str, got {type(text).__name__}")
        if not text:
            continue
        if budget is not None:
            text = text[:budget]
            budget -= len(text)
        text = normalize_text(text)
        if _SURROGATE_RE.search(text):
            text = _replace_surrogates(text)
        yield text


def train_tokenizer(texts: Iterable[str], vocab_size: int, path: Optional[PathLike] = None,
                    split_digits: bool = True, min_frequency: int = 2, *,
                    max_chars: Optional[int] = None, show_progress: bool = False) -> MoonlightTokenizer:
    """Train a byte-level BPE tokenizer with Hugging Face ``tokenizers``' ``BpeTrainer``.

    Args:
        texts: iterable of documents (streamed; normalized with :func:`normalize_text`).
        vocab_size: target vocabulary INCLUDING the 32 specials and 256 byte tokens (>= 288). The
            result can be smaller if the corpus has too few mergeable pairs, never larger.
        path: if given, the tokenizer JSON is saved there (atomic write).
        split_digits: single-digit pre-tokens (True) or groups of up to 3 digits.
        min_frequency: minimum pair count for a merge.
        max_chars: optional cap on the total number of characters consumed from ``texts``.
        show_progress: show the Rust progress bar.

    Special tokens get ids 0..31 (verified), the full 256-symbol byte alphabet is always included,
    so the result round-trips any text.
    """
    vocab_size = int(vocab_size)
    if vocab_size < MIN_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be >= {MIN_VOCAB_SIZE} (32 special + 256 byte tokens), "
                         f"got {vocab_size}")
    if int(min_frequency) < 0:
        raise ValueError(f"min_frequency must be >= 0, got {min_frequency}")
    tk = _tokenizers()
    backend = tk.Tokenizer(tk.models.BPE())
    _configure_backend(backend, split_digits)
    trainer = tk.trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=int(min_frequency),
        special_tokens=_special_added_tokens(),
        initial_alphabet=tk.pre_tokenizers.ByteLevel.alphabet(),
        show_progress=bool(show_progress),
    )
    backend.train_from_iterator(_training_stream(texts, max_chars), trainer=trainer)
    for i, tok in enumerate(SPECIAL_TOKENS):
        got = backend.token_to_id(tok)
        if got != i:
            raise RuntimeError(f"BPE training assigned id {got} to special token {tok} (expected {i})")
    tokenizer = MoonlightTokenizer(backend)
    if path is not None:
        tokenizer.save(path)
    return tokenizer
