"""Deterministic response checkers shared by synthetic data, GRPO rewards and benchmarks (SPEC §13.2).

A *constraint* is a plain dict ``{"type": <name>, **params}``. :func:`check` returns a bool and
:func:`check_all` returns ``(n_passed, n_total, details)``. Checks run on the FINAL answer: when
the response contains ``<|/think|>`` only the text after the last one is checked (reasoning is
not graded).

Supported types and parameters (aliases in brackets):

=====================  =====================================================================
``word_count_max``     ``n`` [max, value, count] - at most n words
``word_count_min``     ``n`` [min, value, count] - at least n words
``sentence_count_max`` ``n`` - at most n sentences
``sentence_count_min`` ``n`` - at least n sentences
``bullet_count``       ``n`` - exactly n lines starting with ``- ``, ``* `` or ``N. ``/``N) ``
``paragraph_count``    ``n`` - exactly n paragraphs (split on ``***`` lines if present, else blank lines)
``include_keywords``   ``keywords`` (list) - all present, case-insensitive, whole-word
``exclude_keywords``   ``keywords`` [forbidden_words] - none present
``keyword_frequency``  ``word`` [keyword], ``at_least`` [n, frequency], optional ``relation``
                       (``"at least"`` default / ``"less than"``)
``json_format``        - whole response parses as JSON (Markdown code fences allowed)
``lowercase``          - no uppercase letters (and some letters)
``uppercase``          - no lowercase letters (and some letters)
``no_commas``          - no ``,`` characters
``ends_with``          ``phrase`` [end_phrase] - stripped response ends with phrase (case-insensitive)
``starts_with``        ``phrase`` - stripped response starts with phrase (case-insensitive)
``title_brackets``     - contains a title wrapped like ``<<title>>``
``sections``           ``n``, optional ``marker`` ("SECTION") - at least n "SECTION X" parts
``language_uz``        - >= 85% of letters Latin, < 10% Cyrillic, few English stop words
                       (``max_english``, default 0.2); ``require_markers`` (default False)
                       additionally demands Uzbek markers (o'/g'/sh/ch or common Uzbek words)
``quotation_wrapped``  - whole response wrapped in double quotes
``max_chars``          ``n`` - at most n characters (stripped)
=====================  =====================================================================

Reward helpers: :func:`reward_exact_number`, :func:`reward_constraints`, :func:`reward_code`,
:func:`compute_reward` (dispatch by ``reward_type``) and :func:`extract_final_answer`.
Pure python; ``reward_code`` imports ``moonlight.benchmarks.execution`` lazily.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Constraint", "CONSTRAINT_TYPES", "check", "check_all", "ifeval_to_constraints", "IFEVAL_SUPPORTED",
    "reward_exact_number", "reward_constraints", "reward_code", "compute_reward", "extract_final_answer",
    "extract_number", "extract_code", "count_words", "count_sentences", "count_bullets", "count_paragraphs",
    "strip_think", "describe",
]

Constraint = Dict[str, Any]

THINK_END = "<|/think|>"
THINK = "<|think|>"

_WORD_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])[\"'»)\]]*\s+|\n+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]\s+|\d{1,3}[.)]\s+)\S")
_FENCE_RE = re.compile(r"^\s*```[\w+-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")

_EN_STOP = frozenset("""the a an and or of to in is are was were be been this that these those it its with for on
as by from at which who what when where how not but have has had will would can could should do does
did i you he she we they them his her our their my your there here than then so if""".split())
_UZ_MARKERS_RE = re.compile(r"[oOgG]'|\b(?:va|bu|uchun|bilan|ham|emas|bo'l\w*|qil\w*|edi|esa|yoki|kerak|"
                            r"mumkin|hamda|lekin|chunki)\b", re.IGNORECASE)


# =============================================================================================== text stats

def strip_think(text: str) -> str:
    """Text after the last ``<|/think|>`` (whole text if absent); a dangling ``<|think|>`` block with no
    end marker yields ''."""
    text = text or ""
    if THINK_END in text:
        return text.rsplit(THINK_END, 1)[1].strip()
    if text.lstrip().startswith(THINK):
        return ""
    return text.strip()


def count_words(text: str) -> int:
    """Words = runs of letters/digits (apostrophes and hyphens inside a word allowed)."""
    return len(_WORD_RE.findall(text or ""))


def _sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT_RE.split((text or "").strip())
    return [p for p in parts if p and _WORD_RE.search(p)]


def count_sentences(text: str) -> int:
    """Sentences: split after ``. ! ? …`` followed by whitespace, and at line breaks; segments with at
    least one word are counted."""
    return len(_sentences(text))


def count_bullets(text: str) -> int:
    """Lines starting with ``- ``, ``* ``, ``• `` or ``N. ``/``N) ``."""
    return sum(1 for line in (text or "").split("\n") if _BULLET_RE.match(line))


def count_paragraphs(text: str) -> int:
    """Paragraphs separated by ``***`` divider lines (IFEval style) if any, else by blank lines."""
    text = (text or "").strip()
    if not text:
        return 0
    if re.search(r"^\s*\*\*\*\s*$", text, re.MULTILINE):
        parts = re.split(r"^\s*\*\*\*\s*$", text, flags=re.MULTILINE)
    else:
        parts = re.split(r"\n\s*\n", text)
    return sum(1 for p in parts if p.strip())


def _kw_pattern(word: str) -> "re.Pattern[str]":
    return re.compile(r"(?<![\w'])" + re.escape(word.strip()) + r"(?![\w'])", re.IGNORECASE)


def _kw_count(text: str, word: str) -> int:
    if not word or not word.strip():
        return 0
    return len(_kw_pattern(word).findall(text))


def _json_ok(text: str) -> bool:
    t = text.strip()
    m = _FENCE_RE.match(t)
    if m:
        t = m.group(1).strip()
    if not t:
        return False
    try:
        json.loads(t)
    except ValueError:
        return False
    return True


def _letters(text: str) -> List[str]:
    return [ch for ch in text if ch.isalpha()]


def _is_uz(text: str, max_english: float = 0.2, require_markers: bool = False) -> bool:
    letters = _letters(text)
    if not letters:
        return False
    latin = sum(1 for ch in letters if ch < "ɐ")
    cyr = sum(1 for ch in letters if "Ѐ" <= ch <= "ӿ")
    if latin / len(letters) < 0.85 or cyr / len(letters) > 0.10:
        return False
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if words and sum(1 for w in words if w in _EN_STOP) / len(words) > max_english:
        return False
    if require_markers and not _UZ_MARKERS_RE.search(text.replace("’", "'").replace("ʻ", "'")):
        return False
    return True


# =============================================================================================== check

def _p(c: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for n in names:
        if n in c and c[n] is not None:
            return c[n]
    return default


def _int(c: Mapping[str, Any], *names: str) -> int:
    v = _p(c, *names)
    if v is None:
        raise ValueError(f"constraint {c.get('type')!r} needs one of {names}")
    return int(v)


def _kw_list(c: Mapping[str, Any], *names: str) -> List[str]:
    v = _p(c, *names, default=[])
    if isinstance(v, str):
        v = [v]
    return [str(x) for x in v if str(x).strip()]


def _check_raw(text: str, c: Mapping[str, Any]) -> bool:
    t = str(c.get("type", ""))
    s = text.strip()
    if t == "word_count_max":
        return count_words(s) <= _int(c, "n", "max", "value", "count")
    if t == "word_count_min":
        return count_words(s) >= _int(c, "n", "min", "value", "count")
    if t == "sentence_count_max":
        return count_sentences(s) <= _int(c, "n", "max", "value", "count")
    if t == "sentence_count_min":
        return count_sentences(s) >= _int(c, "n", "min", "value", "count")
    if t == "bullet_count":
        return count_bullets(s) == _int(c, "n", "value", "count")
    if t == "paragraph_count":
        return count_paragraphs(s) == _int(c, "n", "value", "count")
    if t == "include_keywords":
        kws = _kw_list(c, "keywords", "words")
        return bool(kws) and all(_kw_count(s, k) > 0 for k in kws)
    if t == "exclude_keywords":
        return all(_kw_count(s, k) == 0 for k in _kw_list(c, "keywords", "forbidden_words", "words"))
    if t == "keyword_frequency":
        word = str(_p(c, "word", "keyword", default=""))
        n = int(_p(c, "at_least", "n", "frequency", default=1))
        rel = str(_p(c, "relation", default="at least")).lower()
        cnt = _kw_count(s, word)
        return cnt < n if rel.startswith("less") else cnt >= n
    if t == "json_format":
        return _json_ok(s)
    if t == "lowercase":
        return any(ch.isalpha() for ch in s) and s == s.lower()
    if t == "uppercase":
        return any(ch.isalpha() for ch in s) and s == s.upper()
    if t == "no_commas":
        return "," not in s and "，" not in s
    if t == "ends_with":
        phrase = str(_p(c, "phrase", "end_phrase", "text", default="")).strip()
        return bool(phrase) and s.lower().endswith(phrase.lower())
    if t == "starts_with":
        phrase = str(_p(c, "phrase", "start_phrase", "text", default="")).strip()
        return bool(phrase) and s.lower().startswith(phrase.lower())
    if t == "title_brackets":
        return re.search(r"<<[^\n<>]*[^\s<>][^\n<>]*>>", s) is not None
    if t == "sections":
        marker = str(_p(c, "marker", "section_spliter", "splitter", default="SECTION")).strip()
        n = _int(c, "n", "num_sections", "count")
        found = re.findall(r"(?:^|\s)" + re.escape(marker) + r"\s?\d+", s)
        return len(found) >= n
    if t == "language_uz":
        return _is_uz(s, float(_p(c, "max_english", default=0.2)), bool(_p(c, "require_markers", default=False)))
    if t == "quotation_wrapped":
        return len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "“" and s[-1] == "”"))
    if t == "max_chars":
        return len(s) <= _int(c, "n", "max", "value", "count")
    raise ValueError(f"unknown constraint type {t!r}; supported: {', '.join(CONSTRAINT_TYPES)}")


CONSTRAINT_TYPES: Tuple[str, ...] = (
    "word_count_max", "word_count_min", "sentence_count_max", "sentence_count_min", "bullet_count",
    "paragraph_count", "include_keywords", "exclude_keywords", "keyword_frequency", "json_format",
    "lowercase", "uppercase", "no_commas", "ends_with", "starts_with", "title_brackets", "sections",
    "language_uz", "quotation_wrapped", "max_chars",
)


def check(response: str, constraint: Mapping[str, Any]) -> bool:
    """True iff ``response`` (its final answer, see :func:`strip_think`) satisfies ``constraint``.

    Raises ``ValueError`` for unknown constraint types or missing required parameters.
    """
    return _check_raw(strip_think(response or ""), constraint)


def check_all(response: str, constraints: Sequence[Mapping[str, Any]]) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Check every constraint: ``(n_passed, n_total, details)`` with
    ``details = [{"type": ..., "passed": bool, "constraint": {...}}, ...]``."""
    final = strip_think(response or "")
    details = []
    passed = 0
    for c in constraints:
        ok = bool(_check_raw(final, c))
        passed += ok
        details.append({"type": c.get("type"), "passed": ok, "constraint": dict(c)})
    return passed, len(constraints), details


def describe(constraint: Mapping[str, Any], lang: str = "en") -> str:
    """Natural-language instruction for a constraint (``lang`` ``"uz"`` or ``"en"``)."""
    c = constraint
    t = c.get("type")
    uz = lang == "uz"

    def n(*names: str) -> Any:
        return _p(c, *names, "n", "value", "count")

    if t == "word_count_max":
        return f"Javob {n('max')} ta so'zdan oshmasin." if uz else f"Use at most {n('max')} words."
    if t == "word_count_min":
        return f"Javob kamida {n('min')} ta so'zdan iborat bo'lsin." if uz else f"Use at least {n('min')} words."
    if t == "sentence_count_max":
        return f"Ko'pi bilan {n('max')} ta gap yozing." if uz else f"Write at most {n('max')} sentences."
    if t == "sentence_count_min":
        return f"Kamida {n('min')} ta gap yozing." if uz else f"Write at least {n('min')} sentences."
    if t == "bullet_count":
        return (f"Javobni aynan {n()} ta \"- \" bilan boshlanadigan band ko'rinishida yozing." if uz
                else f"Answer with exactly {n()} bullet points starting with \"- \".")
    if t == "paragraph_count":
        return (f"Aynan {n()} ta paragraf yozing va ularni *** bilan ajrating." if uz
                else f"Write exactly {n()} paragraphs separated by the divider ***.")
    if t == "include_keywords":
        kws = ", ".join(f"\"{k}\"" for k in _kw_list(c, "keywords", "words"))
        return f"Javobda quyidagi so'zlar albatta bo'lsin: {kws}." if uz else f"Include the keywords {kws}."
    if t == "exclude_keywords":
        kws = ", ".join(f"\"{k}\"" for k in _kw_list(c, "keywords", "forbidden_words", "words"))
        return f"Quyidagi so'zlarni ishlatmang: {kws}." if uz else f"Do not use the words {kws}."
    if t == "keyword_frequency":
        w, k = _p(c, "word", "keyword"), _p(c, "at_least", "n", "frequency")
        return (f"\"{w}\" so'zi kamida {k} marta qatnashsin." if uz
                else f"The word \"{w}\" should appear at least {k} times.")
    if t == "json_format":
        return "Butun javobni JSON formatida bering." if uz else "Wrap your entire answer in valid JSON."
    if t == "lowercase":
        return "Butun javobni faqat kichik harflarda yozing." if uz else "Write your entire answer in lowercase letters only."
    if t == "uppercase":
        return "Butun javobni faqat katta harflarda yozing." if uz else "Write your entire answer in capital letters only."
    if t == "no_commas":
        return "Javobda vergul ishlatmang." if uz else "Do not use any commas."
    if t == "ends_with":
        p = _p(c, "phrase", "end_phrase", "text")
        return f"Javobni aynan \"{p}\" bilan tugating." if uz else f"End your answer with the exact phrase \"{p}\"."
    if t == "starts_with":
        p = _p(c, "phrase", "start_phrase", "text")
        return f"Javobni \"{p}\" bilan boshlang." if uz else f"Start your answer with \"{p}\"."
    if t == "title_brackets":
        return ("Javobga <<sarlavha>> ko'rinishidagi sarlavha qo'ying." if uz
                else "Give your answer a title wrapped in double angular brackets, like <<title>>.")
    if t == "sections":
        m = _p(c, "marker", "section_spliter", default="SECTION")
        return (f"Javobni {n('num_sections')} ta bo'limga ajrating, har birini \"{m} X\" bilan belgilang." if uz
                else f"Divide your answer into {n('num_sections')} sections, each marked \"{m} X\".")
    if t == "language_uz":
        return "Faqat o'zbek tilida (lotin yozuvida) javob bering." if uz else "Answer only in Uzbek (Latin script)."
    if t == "quotation_wrapped":
        return "Butun javobni qo'shtirnoq ichiga oling." if uz else "Wrap your entire response in double quotation marks."
    if t == "max_chars":
        return f"Javob {n('max')} belgidan oshmasin." if uz else f"Keep the answer under {n('max')} characters."
    raise ValueError(f"unknown constraint type {t!r}")


# =============================================================================================== IFEval

IFEVAL_SUPPORTED: Tuple[str, ...] = (
    "length_constraints:number_words", "length_constraints:number_sentences",
    "length_constraints:number_paragraphs", "keywords:existence", "keywords:frequency",
    "keywords:forbidden_words", "detectable_format:json_format", "detectable_format:number_bullet_lists",
    "detectable_format:title", "detectable_format:multiple_sections", "change_case:english_lowercase",
    "change_case:english_capital", "punctuation:no_comma", "startend:end_checker", "startend:quotation",
)


def _relation_pair(kind: str, relation: Any, n: int) -> Constraint:
    rel = str(relation or "at least").strip().lower()
    if rel.startswith("less"):
        return {"type": f"{kind}_max", "n": max(0, n - 1)}
    return {"type": f"{kind}_min", "n": n}


def ifeval_to_constraints(instruction_id_list: Sequence[str], kwargs: Optional[Sequence[Any]] = None
                          ) -> Optional[List[Constraint]]:
    """Map an IFEval example to constraints; None if ANY instruction id is unsupported.

    ``kwargs`` is the per-instruction list of argument dicts (keys with None values are ignored, as
    in the HF version where every row carries all keys).
    """
    ids = list(instruction_id_list or [])
    kws: List[Mapping[str, Any]] = list(kwargs or [])
    out: List[Constraint] = []
    for i, iid in enumerate(ids):
        raw = kws[i] if i < len(kws) and isinstance(kws[i], Mapping) else {}
        a = {k: v for k, v in raw.items() if v is not None}
        try:
            if iid == "length_constraints:number_words":
                out.append(_relation_pair("word_count", a.get("relation"), int(a["num_words"])))
            elif iid == "length_constraints:number_sentences":
                out.append(_relation_pair("sentence_count", a.get("relation"), int(a["num_sentences"])))
            elif iid == "length_constraints:number_paragraphs":
                out.append({"type": "paragraph_count", "n": int(a["num_paragraphs"])})
            elif iid == "keywords:existence":
                out.append({"type": "include_keywords", "keywords": list(a["keywords"])})
            elif iid == "keywords:frequency":
                out.append({"type": "keyword_frequency", "word": str(a["keyword"]),
                            "at_least": int(a["frequency"]), "relation": a.get("relation", "at least")})
            elif iid == "keywords:forbidden_words":
                out.append({"type": "exclude_keywords", "keywords": list(a["forbidden_words"])})
            elif iid == "detectable_format:json_format":
                out.append({"type": "json_format"})
            elif iid == "detectable_format:number_bullet_lists":
                out.append({"type": "bullet_count", "n": int(a["num_bullets"])})
            elif iid == "detectable_format:title":
                out.append({"type": "title_brackets"})
            elif iid == "detectable_format:multiple_sections":
                out.append({"type": "sections", "n": int(a["num_sections"]),
                            "marker": str(a.get("section_spliter") or "SECTION")})
            elif iid == "change_case:english_lowercase":
                out.append({"type": "lowercase"})
            elif iid == "change_case:english_capital":
                out.append({"type": "uppercase"})
            elif iid == "punctuation:no_comma":
                out.append({"type": "no_commas"})
            elif iid == "startend:end_checker":
                out.append({"type": "ends_with", "phrase": str(a["end_phrase"])})
            elif iid == "startend:quotation":
                out.append({"type": "quotation_wrapped"})
            else:
                return None
        except (KeyError, TypeError, ValueError):
            return None
    return out


# =============================================================================================== rewards

def extract_final_answer(text: str) -> str:
    """Final answer text: the part after the last ``<|/think|>`` (if any); if it contains a line starting
    with ``Javob:`` / ``Answer:`` / ``Final answer:``, the content of the LAST such line."""
    final = strip_think(text or "")
    found = None
    for line in final.split("\n"):
        m = re.match(r"\s*(?:\*\*)?(?:final answer|answer|javob|yakuniy javob)\s*(?:\*\*)?\s*[:：]\s*(.*)",
                     line, re.IGNORECASE)
        if m:
            found = m.group(1).strip().strip("*").strip()
    return found if found is not None else final


def extract_number(text: str) -> Optional[float]:
    """The LAST number in ``text`` (``1,234`` thousands separators and ``-``, decimals, exponents ok)."""
    found = _NUMBER_RE.findall(text or "")
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def reward_exact_number(response: str, target: Any, rel_tol: float = 1e-6) -> float:
    """1.0 if the last number of the final answer equals ``target`` (relative tol 1e-6), else 0.0."""
    try:
        tgt = float(str(target).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0
    got = extract_number(extract_final_answer(response))
    if got is None:
        return 0.0
    return 1.0 if math.isclose(got, tgt, rel_tol=rel_tol, abs_tol=1e-9) else 0.0


def reward_constraints(response: str, constraints: Sequence[Mapping[str, Any]]) -> float:
    """Fraction of constraints satisfied (1.0 only when all are); 0.0 for an empty list."""
    if not constraints:
        return 0.0
    passed, total, _ = check_all(response, constraints)
    return passed / total


def extract_code(text: str) -> str:
    """Python code from a response: concatenation of ```python / ``` fenced blocks (after the think
    block), or the whole final answer when there is no fence."""
    final = strip_think(text or "")
    blocks = re.findall(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", final, re.DOTALL | re.IGNORECASE)
    if blocks:
        return "\n\n".join(b.strip("\n") for b in blocks)
    return final


def _run_python(code: str, timeout: float) -> Tuple[bool, str, str]:
    try:
        from ..benchmarks.execution import run_python  # lazy: benchmarks package is optional here
    except Exception:  # pragma: no cover - fallback when the benchmarks package is unavailable
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            try:
                proc = subprocess.run([sys.executable, "-I", "-c", code], cwd=d, capture_output=True,
                                      text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return False, "", "timeout"
            return proc.returncode == 0, proc.stdout, proc.stderr
    ok, out, err = run_python(code, timeout=timeout)
    return bool(ok), out, err


def reward_code(response: str, tests: Any, entry_point: Optional[str] = None, timeout: float = 5.0) -> float:
    """1.0 if the code in ``response`` passes ``tests`` when executed (sandboxed subprocess), else 0.0.

    ``tests``: a string of test code (asserts or a HumanEval-style ``def check(candidate)``) or a list
    of assert strings. When the tests define ``check`` and ``entry_point`` is given,
    ``check(<entry_point>)`` is appended.
    """
    code = extract_code(response)
    if not code.strip():
        return 0.0
    if entry_point and not re.search(r"^\s*(?:async\s+)?def\s+" + re.escape(entry_point) + r"\b|^\s*class\s+"
                                     + re.escape(entry_point) + r"\b|^\s*" + re.escape(entry_point) + r"\s*=",
                                     code, re.MULTILINE):
        return 0.0
    test_src = "\n".join(tests) if isinstance(tests, (list, tuple)) else str(tests or "")
    program = code + "\n\n" + test_src + "\n"
    if entry_point and re.search(r"^def check\(", test_src, re.MULTILINE):
        program += f"\ncheck({entry_point})\n"
    ok, _, _ = _run_python(program, timeout)
    return 1.0 if ok else 0.0


def compute_reward(response: str, reward_type: str, reward_data: Mapping[str, Any]) -> float:
    """Dispatch on ``reward_type`` as produced by ``synthetic.grpo_prompt_set``:
    ``exact_number`` (``target``), ``constraints`` / ``detail`` (``constraints``), ``code`` (``tests``,
    ``entry_point``)."""
    if reward_type == "exact_number":
        return reward_exact_number(response, reward_data.get("target"))
    if reward_type in ("constraints", "detail"):
        return reward_constraints(response, reward_data.get("constraints") or [])
    if reward_type == "code":
        return reward_code(response, reward_data.get("tests"), reward_data.get("entry_point"),
                           float(reward_data.get("timeout", 5.0)))
    raise ValueError(f"unknown reward_type {reward_type!r}")
