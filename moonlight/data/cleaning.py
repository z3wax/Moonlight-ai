"""Text / code cleaning and quality filters for pretraining data (SPEC §6).

Pure python (``re``, ``unicodedata``, ``hashlib``); no torch, no numpy.

* :func:`norm_apostrophes` - Uzbek apostrophe look-alikes (``ʻ ʼ ‘ ’ ´`` and a backtick between two
  letters) -> ASCII ``'`` (same convention as the tokenizer, so ``o‘zbek`` and ``o'zbek`` match).
* :func:`norm_text` - Unicode NFC, invisible characters, odd spaces, control chars, whitespace runs.
* :func:`clean_doc` - document cleanup: URLs / e-mails, page-number lines, de-hyphenation across line
  breaks and joining of OCR-wrapped lines inside paragraphs.
* :func:`good_doc` - quality filter: length, letter ratio, script ratio (Cyrillic share for Uzbek
  Latin text), duplicate-line ratio, word-length sanity.
* :func:`clean_code` - strip long license headers, reject minified / generated / data-like files.
* :func:`dedup_exact` - exact de-duplication by content hash.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Callable, Iterable, Iterator, Optional, TypeVar

__all__ = [
    "norm_apostrophes", "norm_text", "clean_doc", "good_doc", "clean_code", "dedup_exact",
    "cyrillic_ratio", "letter_ratio", "clean_for_lang",
]

T = TypeVar("T")

# ---------------------------------------------------------------------------------------- basics

_APOS_ALWAYS = "ʻʼ‘’´ʹ′"  # ʻ ʼ ‘ ’ ´ ʹ ′
_APOS_RE = re.compile("[" + _APOS_ALWAYS + "]")
_BACKTICK_RE = re.compile(r"(?<=[^\W\d_])`(?=[^\W\d_])")  # o`zbek -> o'zbek (not code fences)

_INVISIBLE_RE = re.compile("[​‌‍⁠﻿­]")  # zero-width, BOM, soft hyphen
_SPACES_RE = re.compile("[   -   　\t\f\v]+")
_CTRL_RE = re.compile("[\u0000-\u0008\u000b\u000e-\u001f\u007f-\u009f]")
_MULTISPACE_RE = re.compile(r"[ ]{2,}")
_TRAILING_RE = re.compile(r"[ ]+\n")
_MANY_NL_RE = re.compile(r"\n{3,}")


def norm_apostrophes(text: str) -> str:
    """Map apostrophe look-alikes to ``'``; a backtick only when it sits between two letters."""
    text = _APOS_RE.sub("'", text)
    return _BACKTICK_RE.sub("'", text)


def norm_text(text: str) -> str:
    """Normalize prose: NFC, apostrophes, drop invisible/control characters, unify spaces, collapse
    blank-line runs to one empty line, strip trailing spaces and the whole string.

    Not meant for code (tabs are turned into spaces); use :func:`clean_code` for code.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", "\n").replace(" ", "\n\n")
    text = _INVISIBLE_RE.sub("", text)
    text = _CTRL_RE.sub("", text)
    text = _SPACES_RE.sub(" ", text)
    text = norm_apostrophes(text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _TRAILING_RE.sub("\n", text)
    text = _MANY_NL_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# ---------------------------------------------------------------------------------------- documents

_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"'()\[\]{}]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_TG_HANDLE_LINE_RE = re.compile(r"^\s*(?:[@#][\w.]+[\s,;|]*)+$")  # lines made only of @handles/#tags
_PAGE_LINE_RE = re.compile(
    r"^\s*(?:[-–—]\s*)?(?:(?:page|bet|sahifa|стр\.?)\s*)?\d{1,4}(?:\s*(?:/|of|dan)\s*\d{1,4})?(?:\s*[-–—])?\s*$",
    re.IGNORECASE)
_HYPHEN_BREAK_RE = re.compile(r"([^\W\d_])-\n([^\W\d_A-ZА-ЯЁ])")  # "so'z-\nlar" -> "so'zlar"
_SENT_END = ".!?:;…\"'»)]"
_LIST_START_RE = re.compile(r"^(?:[-*•–—]\s|\d{1,3}[.)]\s|[a-zA-Z][.)]\s|#)")
_EMPTY_PARENS_RE = re.compile(r"\(\s*\)|\[\s*\]")


def _join_wrapped_lines(text: str) -> str:
    """Join hard-wrapped (OCR / PDF) lines inside paragraphs.

    A line break is replaced by a space when the line does not end with sentence punctuation, both
    lines are non-empty, the next line starts with a lowercase letter or digit and is not a list item.
    """
    lines = text.split("\n")
    if len(lines) < 2:
        return text
    out = [lines[0]]
    for nxt in lines[1:]:
        prev = out[-1]
        p = prev.rstrip()
        n = nxt.lstrip()
        if (p and n and p[-1] not in _SENT_END and not _LIST_START_RE.match(n)
                and (n[0].islower() or n[0].isdigit()) and len(p) >= 30):
            out[-1] = p + " " + n
        else:
            out.append(nxt)
    return "\n".join(out)


def clean_doc(text: str, lang: Optional[str] = None) -> str:
    """Clean one prose document (returns a possibly empty string).

    Steps: :func:`norm_text`; drop URLs and e-mail addresses (and leftover empty brackets); drop
    lines that are only a page number (``12``, ``- 12 -``, ``Page 3 of 10``, ``5-bet``) or only
    ``@handles``/``#hashtags``; de-hyphenate words split across lines; join OCR-wrapped lines; final
    whitespace normalization. ``lang`` is accepted for API symmetry (rules are script-agnostic).
    """
    del lang
    if not text:
        return ""
    text = norm_text(text)
    text = _URL_RE.sub("", text)
    text = _EMAIL_RE.sub("", text)
    text = _EMPTY_PARENS_RE.sub("", text)
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if s and (_PAGE_LINE_RE.match(s) or _TG_HANDLE_LINE_RE.match(s)):
            continue
        if s and re.fullmatch(r"\d{1,4}\s*-\s*bet", s, re.IGNORECASE):
            continue
        kept.append(line)
    text = "\n".join(kept)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _join_wrapped_lines(text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = _MANY_NL_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def letter_ratio(text: str) -> float:
    """Fraction of non-space characters that are letters (0.0 for empty text)."""
    total = letters = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if ch.isalpha():
            letters += 1
    return letters / total if total else 0.0


def cyrillic_ratio(text: str) -> float:
    """Fraction of letters that are Cyrillic (U+0400..U+04FF)."""
    letters = cyr = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if "Ѐ" <= ch <= "ӿ":
                cyr += 1
    return cyr / letters if letters else 0.0


def _latin_ratio(text: str) -> float:
    letters = latin = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if ch < "ɐ":  # Basic Latin .. Latin Extended-B
                latin += 1
    return latin / letters if letters else 0.0


def good_doc(text: str, min_chars: int = 200, lang: Optional[str] = "uz") -> bool:
    """Quality filter for a CLEANED prose document.

    Rejects documents that are shorter than ``min_chars``; whose letter ratio (letters / non-space
    chars) is below 0.55; whose Cyrillic share of letters exceeds 10% for ``lang="uz"`` (Moonlight
    trains on Uzbek Latin) or whose Latin share is below 85% for ``uz``/``en``; where more than 30%
    of the non-empty lines (>= 5 lines) are duplicates; whose mean word length is outside [2, 15];
    or which have fewer than 20 words (this last rule is skipped when ``min_chars < 50``).
    ``lang`` other than ``uz``/``en`` (e.g. None) skips the script checks.
    """
    if not text or len(text) < max(1, int(min_chars)):
        return False
    if letter_ratio(text) < 0.55:
        return False
    if lang in ("uz", "en"):
        if lang == "uz" and cyrillic_ratio(text) > 0.10:
            return False
        if _latin_ratio(text) < 0.85:
            return False
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 5 and 1.0 - len(set(lines)) / len(lines) > 0.30:
        return False
    words = text.split()
    if min_chars >= 50 and len(words) < 20:
        return False
    if words:
        mean_len = sum(len(w) for w in words) / len(words)
        if not 2.0 <= mean_len <= 15.0:
            return False
    return True


def clean_for_lang(text: str, lang: str, min_chars: int = 200) -> Optional[str]:
    """Convenience: ``clean_code`` for ``lang == "code"``, else ``clean_doc`` + ``good_doc``.

    Returns the cleaned text or None when the document is rejected.
    """
    if lang == "code":
        return clean_code(text)
    cleaned = clean_doc(text, lang)
    return cleaned if good_doc(cleaned, min_chars, lang if lang in ("uz", "en") else None) else None


# ---------------------------------------------------------------------------------------- code

_LICENSE_WORDS = re.compile(r"licen[cs]e|copyright|\(c\)|permission is hereby granted|warranty|"
                            r"redistribut|spdx-license", re.IGNORECASE)
_GENERATED_RE = re.compile(r"auto-?generated|generated by|do not edit|automatically generated", re.IGNORECASE)


def _strip_license_header(lines: list) -> list:
    """Remove a leading comment block / module docstring longer than 20 lines that mentions a license."""
    i = 0
    n = len(lines)
    # keep a shebang / encoding line
    keep = []
    while i < n and (lines[i].startswith("#!") or re.match(r"#.*coding[:=]", lines[i])):
        keep.append(lines[i])
        i += 1
    start = i
    if i < n and lines[i].lstrip().startswith("#"):
        while i < n and (lines[i].lstrip().startswith("#") or not lines[i].strip()):
            i += 1
    elif i < n and lines[i].lstrip()[:3] in ('"""', "'''"):
        quote = lines[i].lstrip()[:3]
        rest = lines[i].lstrip()[3:]
        if quote in rest:  # one-line docstring
            return lines
        i += 1
        while i < n and quote not in lines[i]:
            i += 1
        i += 1  # include closing line
    block = lines[start:i]
    if len(block) > 20 and _LICENSE_WORDS.search("\n".join(block)):
        while i < n and not lines[i].strip():
            i += 1
        return keep + lines[i:]
    return lines


def clean_code(text: str, max_line_len: int = 1000, max_mean_line_len: float = 100.0,
               min_alnum: float = 0.25, min_lines: int = 3) -> Optional[str]:
    """Clean a source file; returns None when the file should be dropped.

    * normalizes newlines, strips trailing whitespace and BOM / zero-width characters (tabs kept);
    * removes a leading license header (comment block or docstring) longer than 20 lines;
    * drops files that are auto-generated, have any line longer than ``max_line_len``, a mean line
      length above ``max_mean_line_len`` (minified), an alphanumeric fraction below ``min_alnum``
      (data blobs) or fewer than ``min_lines`` non-empty lines.
    """
    if not text:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE_RE.sub("", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    head = "\n".join(lines[:40])
    if _GENERATED_RE.search(head):
        return None
    lines = _strip_license_header(lines)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    nonempty = [ln for ln in lines if ln.strip()]
    if len(nonempty) < int(min_lines):
        return None
    if any(len(ln) > max_line_len for ln in lines):
        return None
    if sum(len(ln) for ln in nonempty) / len(nonempty) > max_mean_line_len:
        return None
    body = "\n".join(lines)
    stripped = re.sub(r"\s", "", body)
    if not stripped or sum(ch.isalnum() for ch in stripped) / len(stripped) < min_alnum:
        return None
    body = _MANY_NL_RE.sub("\n\n\n", body)  # at most two blank lines (PEP 8 style)
    return body + "\n"


# ---------------------------------------------------------------------------------------- dedup

def dedup_exact(items: Iterable[T], key: Optional[Callable[[T], str]] = None) -> Iterator[T]:
    """Yield items whose text (``key(item)`` or the item itself) was not seen before.

    Uses a set of 16-byte BLAKE2b digests of the whitespace-stripped text (memory ~ 50 B / doc).
    """
    seen = set()
    for item in items:
        text = key(item) if key is not None else item
        h = hashlib.blake2b(str(text).strip().encode("utf-8", "surrogatepass"), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        yield item
