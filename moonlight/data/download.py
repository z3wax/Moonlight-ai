"""Cached HTTP downloads and streaming row readers (SPEC §6).

* :func:`download` - fetch one URL into a cache directory with an ``Authorization: Bearer $HF_TOKEN``
  header (when a token is available), retries with exponential backoff, resume of partial ``.part``
  files (HTTP Range), atomic rename on completion and readable progress logging. A file that is
  already cached is returned immediately without network access.
* :func:`iter_rows` - stream dict rows from ``parquet`` (``pyarrow.ParquetFile.iter_batches``),
  ``jsonl``, ``json`` (top-level list, or JSON lines saved with a ``.json`` suffix) and ``json.gz``.
* :func:`resolve_hf_shards` - list a dataset directory through the Hub tree API; used to recover
  when a fixed shard name in :mod:`moonlight.data.sources` 404s after a dataset was re-sharded.
* :func:`iter_source_rows` - rows of a registered source, downloading shards lazily (only as many
  as the consumer actually reads).

Only the standard library and ``pyarrow`` (lazily, for parquet) are used; torch is never imported.
Environment: ``HF_TOKEN`` (auth), ``HF_HUB_OFFLINE=1`` (never touch the network; cached files only),
``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` (custom CA bundle), standard ``HTTPS_PROXY``.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

from .sources import FORMATS, HF_BASE, DatasetSpec, get_source, hf_url, repo_and_path

__all__ = [
    "DownloadError", "download", "iter_rows", "resolve_hf_shards", "cache_path_for", "infer_format",
    "iter_source_rows", "source_files", "is_offline_env",
]

PathLike = Union[str, "os.PathLike[str]"]

_ENV_TOKEN: Any = object()  # sentinel: read HF_TOKEN at call time
_USER_AGENT = "moonlight-llm/0.1 (+https://huggingface.co)"
_CHUNK = 1 << 20  # 1 MiB
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    """A download failed permanently. ``status`` is the HTTP status (None for network errors)."""

    def __init__(self, message: str, status: Optional[int] = None, url: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.url = url


def _logger() -> Any:
    from ..utils import get_logger  # torch-free
    return get_logger("moonlight.data")


def is_offline_env() -> bool:
    """True when ``HF_HUB_OFFLINE`` / ``MOONLIGHT_OFFLINE`` is set to a truthy value."""
    for key in ("HF_HUB_OFFLINE", "MOONLIGHT_OFFLINE"):
        if os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


def _resolve_token(token: Any) -> Optional[str]:
    if token is _ENV_TOKEN:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token = (token or "").strip()
    return token or None


def _ssl_context() -> ssl.SSLContext:
    cafile = None
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        cand = os.environ.get(key)
        if cand and os.path.isfile(cand):
            cafile = cand
            break
    return ssl.create_default_context(cafile=cafile)


class _AuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects, keeping the bearer token only while the host stays the same (the HF CDN
    serves pre-signed URLs that must not receive our token)."""

    def __init__(self, auth: Optional[str]) -> None:
        super().__init__()
        self._auth = auth

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and self._auth:
            if urllib.parse.urlsplit(newurl).netloc == urllib.parse.urlsplit(req.full_url).netloc:
                new.add_unredirected_header("Authorization", self._auth)
        return new


def _open(url: str, token: Optional[str], headers: Optional[Dict[str, str]] = None, timeout: float = 60.0):
    auth = f"Bearer {token}" if token else None
    opener = urllib.request.build_opener(_AuthRedirectHandler(auth),
                                         urllib.request.HTTPSHandler(context=_ssl_context()))
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **(headers or {})})
    if auth:
        req.add_unredirected_header("Authorization", auth)  # re-added by the handler on same-host hops
    return opener.open(req, timeout=timeout)


def cache_path_for(url: str, dest_dir: PathLike) -> Path:
    """Deterministic cache location of ``url`` inside ``dest_dir``.

    HF dataset files map to ``<dest_dir>/<owner>/<repo>/<path>`` (shard names repeat across repos);
    other URLs to ``<dest_dir>/<host>/<path>``.
    """
    root = Path(os.path.expandvars(os.path.expanduser(str(dest_dir))))
    rp = repo_and_path(url)
    if rp is not None:
        rel = Path(*rp[0].split("/")) / Path(*urllib.parse.unquote(rp[1]).split("/"))
    else:
        parts = urllib.parse.urlsplit(url)
        path = urllib.parse.unquote(parts.path).strip("/") or "index"
        rel = Path(parts.netloc.replace(":", "_")) / Path(*path.split("/"))
    rel_parts = [p for p in rel.parts if p not in ("", ".", "..")]
    return root.joinpath(*rel_parts)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"  # pragma: no cover


class _Progress:
    """Throttled progress logging (every ``every`` seconds and at completion)."""

    def __init__(self, name: str, total: Optional[int], start: int, every: float = 10.0) -> None:
        self.name, self.total, self.done, self.every = name, total, start, every
        self.t0 = self.last = time.monotonic()
        self.start = start

    def update(self, n: int) -> None:
        self.done += n
        now = time.monotonic()
        if now - self.last >= self.every:
            self.last = now
            self.log()

    def log(self, final: bool = False) -> None:
        dt = max(time.monotonic() - self.t0, 1e-6)
        speed = (self.done - self.start) / dt
        if self.total:
            pct = 100.0 * self.done / self.total
            msg = f"{_fmt_bytes(self.done)} / {_fmt_bytes(self.total)} ({pct:.0f}%)"
        else:
            msg = _fmt_bytes(self.done)
        state = "done" if final else "downloading"
        _logger().info("%s %s: %s at %s/s", state, self.name, msg, _fmt_bytes(speed))


def download(url: str, dest_dir: PathLike, token: Any = _ENV_TOKEN, *, retries: int = 5,
             backoff: float = 2.0, timeout: float = 60.0, force: bool = False,
             progress: bool = True) -> Path:
    """Download ``url`` into the cache under ``dest_dir`` and return the local path.

    Args:
        url: http(s) URL (normally an HF ``resolve/main`` URL).
        dest_dir: cache root (``~`` and env vars expanded); see :func:`cache_path_for` for layout.
        token: bearer token; default reads ``HF_TOKEN`` at call time; ``None``/"" -> anonymous.
        retries: attempts for transient failures (network errors, 408/429/5xx); waits
            ``backoff ** attempt`` seconds (capped at 60 s) between attempts.
        timeout: socket timeout per request (seconds).
        force: re-download even when cached.
        progress: log progress every ~10 s.

    The body is streamed to ``<file>.part`` (resumed with an HTTP Range request when a partial file
    exists), checked against ``Content-Length`` and atomically renamed. Raises :class:`DownloadError`
    on permanent failures (4xx other than 408/429, retries exhausted, offline mode with no cache).
    """
    dest = cache_path_for(url, dest_dir)
    if dest.is_file() and not force:
        return dest
    if is_offline_env():
        raise DownloadError(f"offline mode (HF_HUB_OFFLINE) and not cached: {url}", None, url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    tok = _resolve_token(token)
    attempts = max(1, int(retries))
    last_err: Optional[BaseException] = None
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have > 0 else {}
        try:
            with _open(url, tok, headers, timeout) as resp:
                status = getattr(resp, "status", 200)
                if have > 0 and status != 206:  # server ignored Range -> restart from scratch
                    have = 0
                length = resp.headers.get("Content-Length")
                total = (int(length) + have) if (length and length.isdigit()) else None
                if have > 0 and status == 206:
                    crange = resp.headers.get("Content-Range", "")  # "bytes a-b/total"
                    if "/" in crange and crange.rsplit("/", 1)[1].isdigit():
                        total = int(crange.rsplit("/", 1)[1])
                prog = _Progress(dest.name, total, have) if progress else None
                with open(part, "ab" if have > 0 else "wb") as fh:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        if prog is not None:
                            prog.update(len(chunk))
                    fh.flush()
                    os.fsync(fh.fileno())
            size = part.stat().st_size
            if total is not None and size != total:
                raise IOError(f"incomplete download: got {size} of {total} bytes")
            os.replace(part, dest)
            if prog is not None:
                prog.log(final=True)
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():  # range not satisfiable: stale partial file
                part.unlink()
                last_err = e
                continue
            if e.code not in _RETRY_STATUS:
                hint = ""
                if e.code in (401, 403):
                    hint = " (gated or private dataset? accept its terms on the Hub and set HF_TOKEN)"
                raise DownloadError(f"HTTP {e.code} for {url}{hint}", e.code, url) from None
            last_err = e
        except urllib.error.URLError as e:
            m = re.search(r"Tunnel connection failed: (\d{3})", str(e.reason))
            if m and int(m.group(1)) in (403, 407):  # proxy refuses this host: retrying cannot help
                raise DownloadError(f"proxy refused {url} (HTTP {m.group(1)})", int(m.group(1)), url) from None
            last_err = e
        except (OSError, ConnectionError, TimeoutError) as e:
            last_err = e
        if attempt + 1 < attempts:
            wait = min(60.0, float(backoff) ** (attempt + 1))
            _logger().warning("download %s failed (%s); retry %d/%d in %.0fs", url, last_err,
                              attempt + 1, attempts - 1, wait)
            time.sleep(wait)
    status = getattr(last_err, "code", None)
    raise DownloadError(f"download failed after {attempts} attempts: {url} ({last_err})", status, url)


# --------------------------------------------------------------------------------------------- rows

def infer_format(path: PathLike) -> str:
    """File format from the suffix (``.json.gz`` / ``.jsonl.gz`` -> ``json.gz``)."""
    name = str(path).lower()
    if name.endswith((".json.gz", ".jsonl.gz")):
        return "json.gz"
    for fmt in ("parquet", "jsonl", "json"):
        if name.endswith("." + fmt):
            return fmt
    raise ValueError(f"cannot infer data format of {path}; expected one of {FORMATS}")


def _project(row: Any, columns: Optional[Sequence[str]]) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    if columns is None:
        return row
    return {c: row.get(c) for c in columns}


def _iter_json_lines(fh: Iterable[str], columns: Optional[Sequence[str]], where: str) -> Iterator[Dict[str, Any]]:
    bad = 0
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            bad += 1
            continue
        row = _project(obj, columns)
        if row is not None:
            yield row
    if bad:
        _logger().warning("%s: skipped %d malformed JSON line(s)", where, bad)


def _iter_json_text(fh: io.TextIOBase, columns: Optional[Sequence[str]], where: str) -> Iterator[Dict[str, Any]]:
    """Rows of a JSON document that is either a top-level list/object or JSON lines."""
    head = ""
    while True:  # peek the first non-space character
        ch = fh.read(1)
        if not ch:
            return
        if not ch.isspace():
            head = ch
            break
    if head == "[":
        data = json.loads(head + fh.read())  # whole list in memory (json has no streaming list reader)
        for obj in data:
            row = _project(obj, columns)
            if row is not None:
                yield row
        return
    first = head + fh.readline()
    try:
        obj = json.loads(first)
    except ValueError:
        obj = None
    if isinstance(obj, dict):  # JSON lines
        row = _project(obj, columns)
        if row is not None:
            yield row
        yield from _iter_json_lines(fh, columns, where)
        return
    # a single (pretty-printed) top-level object
    data = json.loads(first + fh.read())
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        items = lists[0] if len(lists) == 1 else [data]
        for obj in items:
            row = _project(obj, columns)
            if row is not None:
                yield row


def iter_rows(path: PathLike, fmt: Optional[str] = None, columns: Optional[Sequence[str]] = None,
              batch_size: int = 256) -> Iterator[Dict[str, Any]]:
    """Stream rows (dicts) from a local data file.

    Args:
        path: local file.
        fmt: ``parquet`` | ``jsonl`` | ``json`` | ``json.gz`` (None -> inferred from the suffix).
        columns: subset of columns to return (parquet reads only those that exist in the file;
            JSON rows are projected, missing keys -> None). None -> all columns.
        batch_size: parquet record-batch size.

    ``json`` handles a top-level list (loaded with ``json.load``, the only option for a JSON array),
    a single object, or JSON lines saved with a ``.json`` suffix; ``json.gz`` is the same inside gzip.
    Malformed JSON lines are skipped with a warning.
    """
    path = Path(path)
    fmt = fmt or infer_format(path)
    if fmt == "parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(path))
        cols = None
        if columns is not None:
            names = set(pf.schema_arrow.names)
            cols = [c for c in columns if c in names] or None
        for batch in pf.iter_batches(batch_size=max(1, int(batch_size)), columns=cols):
            for row in batch.to_pylist():
                yield row if columns is None else {c: row.get(c) for c in columns}
    elif fmt == "jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield from _iter_json_lines(fh, columns, str(path))
    elif fmt == "json":
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield from _iter_json_text(fh, columns, str(path))
    elif fmt == "json.gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            yield from _iter_json_text(fh, columns, str(path))
    else:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")


# --------------------------------------------------------------------------------------------- HF API

def resolve_hf_shards(repo: str, dir: str, token: Any = _ENV_TOKEN,
                      suffixes: Sequence[str] = (".parquet", ".jsonl", ".json", ".json.gz", ".jsonl.gz"),
                      timeout: float = 30.0) -> List[str]:
    """List data files of ``<repo>/<dir>`` via ``https://huggingface.co/api/datasets/<repo>/tree/main/<dir>``.

    Returns sorted ``resolve/main`` URLs of files ending with one of ``suffixes`` (follows the
    ``Link: rel="next"`` pagination). Raises :class:`DownloadError` when the listing fails.
    """
    if is_offline_env():
        raise DownloadError("offline mode (HF_HUB_OFFLINE): cannot list shards", None, repo)
    tok = _resolve_token(token)
    dir = dir.strip("/")
    url: Optional[str] = f"{HF_BASE}/api/datasets/{repo}/tree/main/{urllib.parse.quote(dir)}"
    paths: List[str] = []
    while url:
        try:
            with _open(url, tok, {"Accept": "application/json"}, timeout) as resp:
                entries = json.loads(resp.read().decode("utf-8"))
                link = resp.headers.get("Link", "") or ""
        except urllib.error.HTTPError as e:
            raise DownloadError(f"HTTP {e.code} listing {repo}/{dir}", e.code, url) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise DownloadError(f"cannot list {repo}/{dir}: {e}", None, url) from None
        for ent in entries if isinstance(entries, list) else []:
            if isinstance(ent, dict) and ent.get("type") == "file":
                p = str(ent.get("path", ""))
                if p.endswith(tuple(suffixes)):
                    paths.append(p)
        url = None
        for part in link.split(","):
            if 'rel="next"' in part and "<" in part and ">" in part:
                url = part[part.index("<") + 1:part.index(">")]
    return [hf_url(repo, p) for p in sorted(set(paths))]


# --------------------------------------------------------------------------------------------- sources

def _spec(source: Union[str, DatasetSpec]) -> DatasetSpec:
    return source if isinstance(source, DatasetSpec) else get_source(source)


def source_files(source: Union[str, DatasetSpec], cache_dir: PathLike, max_shards: Optional[int] = None,
                 token: Any = _ENV_TOKEN) -> Iterator[Path]:
    """Yield local paths of a source's shards, downloading each one only when it is requested.

    If a fixed shard URL 404s, the shard's directory is listed with :func:`resolve_hf_shards` and the
    remaining shards are replaced by the listing (files with the same stem prefix, e.g. ``train-``).
    """
    spec = _spec(source)
    urls = list(spec.urls)
    seen = set()
    i = 0
    resolved = False
    n_yield = 0
    while i < len(urls):
        if max_shards is not None and n_yield >= max_shards:
            return
        url = urls[i]
        i += 1
        if url in seen:
            continue
        seen.add(url)
        try:
            path = download(url, cache_dir, token)
        except DownloadError as e:
            rp = repo_and_path(url)
            if e.status == 404 and rp is not None and not resolved:
                resolved = True
                d = rp[1].rsplit("/", 1)[0] if "/" in rp[1] else ""
                stem = rp[1].rsplit("/", 1)[-1].split("-")[0]
                listed = [u for u in resolve_hf_shards(rp[0], d, token)
                          if repo_and_path(u)[1].rsplit("/", 1)[-1].startswith(stem)]  # type: ignore[index]
                if listed:
                    _logger().warning("%s: %s not found; using %d listed shard(s) of %s/%s", spec.name,
                                      rp[1], len(listed), rp[0], d)
                    urls = urls[:i - 1] + [u for u in listed if u not in seen]
                    i -= 1
                    continue
            raise
        n_yield += 1
        yield path


def iter_source_rows(source: Union[str, DatasetSpec], cache_dir: PathLike, columns: Optional[Sequence[str]] = None,
                     max_rows: Optional[int] = None, max_shards: Optional[int] = None,
                     token: Any = _ENV_TOKEN, batch_size: int = 256) -> Iterator[Dict[str, Any]]:
    """Stream raw rows of a registered source across its shards (lazy shard downloads).

    ``columns`` defaults to ``spec.columns``. Stops after ``max_rows`` rows when given.
    """
    spec = _spec(source)
    cols = spec.columns if columns is None else columns
    n = 0
    for path in source_files(spec, cache_dir, max_shards=max_shards, token=token):
        for row in iter_rows(path, spec.fmt, columns=cols, batch_size=batch_size):
            if max_rows is not None and n >= max_rows:
                return
            n += 1
            yield row
        if max_rows is not None and n >= max_rows:
            return


def _main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - manual utility
    """``python -m moonlight.data.download SOURCE [CACHE_DIR]``: fetch the first shard, print 2 rows."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m moonlight.data.download SOURCE [CACHE_DIR]")
        return 2
    cache = args[1] if len(args) > 1 else "~/.cache/moonlight"
    for i, row in enumerate(iter_source_rows(args[0], cache, max_rows=2, max_shards=1)):
        print(json.dumps(row, ensure_ascii=False, default=str)[:800])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
