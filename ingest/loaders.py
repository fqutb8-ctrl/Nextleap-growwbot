from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import Comment, Declaration, Doctype, ProcessingInstruction

import config
from rag.logging_utils import get_logger
from rag.schemas import Block, Document, Section

log = get_logger("fetch")

_REQUIRED_COLUMNS = ("url", "scheme", "scheme_name", "category")
_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_WHITESPACE = re.compile(config.LOADER_WHITESPACE_PATTERN)
_DROP_TAGS = frozenset(config.LOADER_DROP_TAGS)
_BOILERPLATE_SELECTORS = tuple(config.LOADER_BOILERPLATE_SELECTORS)
_JUNK_TEXTS = frozenset(t.lower() for t in config.LOADER_JUNK_ELEMENT_TEXTS)
_HEADING_TAGS = tuple(config.LOADER_HEADING_TAGS)
_HEADING_LEVEL = {tag: int(tag[1:]) for tag in _HEADING_TAGS}
_STRUCTURAL_TAGS = ("table", "ul", "ol", "dl")
_LIST_TAGS = ("ul", "ol", "dl")
_MAX_HEADING_LEVEL = config.LOADER_MAX_HEADING_LEVEL
_HEADING_PATH_MODE = config.LOADER_HEADING_PATH_MODE
_OVERVIEW_HEADING = config.LOADER_OVERVIEW_HEADING
_RISK_PILL = re.compile(config.LOADER_RISK_PILL_PATTERN)
_KV_MAX_LABEL_CHARS = config.LOADER_KV_MAX_LABEL_CHARS
_KV_REJECT_LABEL = re.compile(config.LOADER_KV_REJECT_LABEL_PATTERN)
_KV_REJECT_NUMERIC_LABEL = re.compile(config.LOADER_KV_REJECT_NUMERIC_LABEL_PATTERN)
_INTERACTIVE_CLASSES = frozenset(config.LOADER_INTERACTIVE_CLASSES)
_COMMENT_TYPES = (Comment, Declaration, Doctype, ProcessingInstruction)


class SourceValidationError(ValueError):
    pass


class FetchError(RuntimeError):
    pass


class StructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceRow:
    url: str
    scheme: str
    scheme_name: str
    category: str


@dataclass
class FetchResult:
    url: str
    scheme: str
    scheme_name: str
    category: str
    status: int
    content_type: str
    n_bytes: int
    fetched_at: str
    elapsed_ms: float
    attempts: int
    cached: bool
    html_path: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _raw_html_path(scheme: str) -> Path:
    return config.RAW_HTML_DIR / f"raw_{scheme}.html"


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in config.ALLOWED_URL_SCHEMES:
        raise SourceValidationError(
            f"scheme {parsed.scheme!r} not allowed for {url!r}; "
            f"allowed={config.ALLOWED_URL_SCHEMES}"
        )
    host = (parsed.hostname or "").lower()
    if host not in config.ALLOWED_HOSTS:
        raise SourceValidationError(
            f"host {host!r} not allowlisted for {url!r}; "
            f"allowed={list(config.ALLOWED_HOSTS)}"
        )
    if not parsed.path or parsed.path == "/":
        raise SourceValidationError(f"url {url!r} has no resource path")


def load_sources(path: str | Path | None = None) -> list[SourceRow]:
    csv_path = Path(path) if path else config.SOURCES_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"sources csv not found: {csv_path}")

    rows: list[SourceRow] = []
    seen_slugs: set[str] = set()
    seen_urls: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SourceValidationError(f"{csv_path} missing columns: {missing}")

        for line_no, raw in enumerate(reader, start=2):
            if raw.get(None):
                raise SourceValidationError(f"{csv_path}:{line_no} has extra fields")
            values = {key: (raw.get(key) or "").strip() for key in _REQUIRED_COLUMNS}
            empty = [key for key, value in values.items() if not value]
            if empty:
                raise SourceValidationError(f"{csv_path}:{line_no} empty fields: {empty}")

            row = SourceRow(**values)
            validate_url(row.url)
            if row.scheme in seen_slugs:
                raise SourceValidationError(f"{csv_path}:{line_no} duplicate scheme {row.scheme!r}")
            if row.url in seen_urls:
                raise SourceValidationError(f"{csv_path}:{line_no} duplicate url {row.url!r}")
            seen_slugs.add(row.scheme)
            seen_urls.add(row.url)
            rows.append(row)

    if not rows:
        raise SourceValidationError(f"{csv_path} contains no source rows")
    return rows


def _build_client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": config.HTTP_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-IN,en;q=0.9",
        },
        timeout=config.HTTP_TIMEOUT_S,
        follow_redirects=True,
    )


def _load_previous_fetched_at(url: str) -> str | None:
    if not config.FETCH_MANIFEST.is_file():
        return None
    try:
        payload = json.loads(config.FETCH_MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    for entry in payload.get("sources", []):
        if entry.get("url") == url:
            return entry.get("fetched_at")
    return None


def _from_cache(row: SourceRow) -> FetchResult | None:
    html_path = _raw_html_path(row.scheme)
    if not html_path.is_file():
        return None
    payload = html_path.read_bytes()
    if not payload.strip():
        return None
    fetched_at = _load_previous_fetched_at(row.url) or datetime.fromtimestamp(
        html_path.stat().st_mtime, timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    return FetchResult(
        url=row.url,
        scheme=row.scheme,
        scheme_name=row.scheme_name,
        category=row.category,
        status=200,
        content_type="text/html",
        n_bytes=len(payload),
        fetched_at=fetched_at,
        elapsed_ms=0.0,
        attempts=0,
        cached=True,
        html_path=str(html_path.relative_to(config.BASE_DIR)),
    )


def fetch_url(row: SourceRow, client: httpx.Client) -> FetchResult:
    validate_url(row.url)
    total_attempts = config.HTTP_MAX_RETRIES + 1
    last_error = ""

    for attempt in range(1, total_attempts + 1):
        started = time.perf_counter()
        try:
            response = client.get(row.url)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < total_attempts:
                delay = config.HTTP_BACKOFF_S * attempt
                log.warn("retry", scheme=row.scheme, attempt=attempt, delay_s=delay, error=last_error)
                time.sleep(delay)
                continue
            raise FetchError(f"{row.url} failed after {total_attempts} attempts: {last_error}") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.status_code in _RETRY_STATUSES and attempt < total_attempts:
            delay = config.HTTP_BACKOFF_S * attempt
            log.warn(
                "retry",
                scheme=row.scheme,
                attempt=attempt,
                status=response.status_code,
                delay_s=delay,
            )
            time.sleep(delay)
            continue

        if response.status_code != 200:
            raise FetchError(
                f"{row.url} returned HTTP {response.status_code} "
                f"(expected 200); aborting ingestion for this source"
            )

        content_type = response.headers.get("content-type", "")
        base_type = content_type.split(";")[0].strip().lower()
        if base_type not in config.HTML_CONTENT_TYPES:
            raise FetchError(
                f"{row.url} returned content-type {content_type!r}; "
                f"expected one of {list(config.HTML_CONTENT_TYPES)}"
            )

        payload = response.content
        if not payload.strip():
            raise FetchError(f"{row.url} returned an empty body")

        html_path = _raw_html_path(row.scheme)
        html_path.write_bytes(payload)

        return FetchResult(
            url=row.url,
            scheme=row.scheme,
            scheme_name=row.scheme_name,
            category=row.category,
            status=response.status_code,
            content_type=content_type,
            n_bytes=len(payload),
            fetched_at=_now_iso(),
            elapsed_ms=elapsed_ms,
            attempts=attempt,
            cached=False,
            html_path=str(html_path.relative_to(config.BASE_DIR)),
        )

    raise FetchError(f"{row.url} exhausted retries: {last_error or 'unknown error'}")


def write_manifest(results: list[FetchResult]) -> Path:
    config.ensure_dirs()
    live = [r for r in results if not r.cached]
    payload = {
        "built_at": _now_iso(),
        "n_sources": len(results),
        "latest_fetched_at": max((r.fetched_at for r in live), default="") or max(
            (r.fetched_at for r in results), default=""
        ),
        "sources": [asdict(r) for r in results],
    }
    config.FETCH_MANIFEST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return config.FETCH_MANIFEST


def fetch_all(use_cache: bool = True, sources: list[SourceRow] | None = None) -> list[FetchResult]:
    rows = sources if sources is not None else load_sources()
    config.ensure_dirs()
    results: list[FetchResult] = []

    with _build_client() as client:
        for index, row in enumerate(rows):
            if index:
                time.sleep(config.HTTP_DELAY_S)

            result = _from_cache(row) if use_cache else None
            if result is None:
                result = fetch_url(row, client)

            results.append(result)
            log.info(
                "load",
                scheme=row.scheme,
                status=result.status,
                bytes=result.n_bytes,
                attempts=result.attempts,
                cached=result.cached,
                ms=result.elapsed_ms,
            )

    manifest_path = write_manifest(results)
    log.info("manifest", path=str(manifest_path.relative_to(config.BASE_DIR)), n=len(results))
    return results


def _soup(markup: str) -> BeautifulSoup:
    return BeautifulSoup(markup, "lxml")


def _norm(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _text_len(node: Tag) -> int:
    return len(node.get_text(" ", strip=True))


def _is_alive(node: Tag) -> bool:
    return not getattr(node, "decomposed", False)


def clean_html(html: str) -> str:
    soup = _soup(html)
    _strip_boilerplate(soup)
    _strip_comments(soup)
    _collapse_whitespace(soup)
    return str(soup)


def _strip_comments(root: Tag) -> None:
    for node in root.find_all(string=lambda s: isinstance(s, _COMMENT_TYPES)):
        node.extract()


def _collapse_whitespace(root: Tag) -> None:
    for node in root.find_all(string=True):
        if isinstance(node, _COMMENT_TYPES):
            continue
        collapsed = _WHITESPACE.sub(" ", str(node))
        if collapsed != str(node):
            node.replace_with(collapsed)


def _strip_boilerplate(root: Tag) -> None:
    for selector in _BOILERPLATE_SELECTORS:
        for node in list(root.select(selector)):
            if _is_alive(node):
                node.decompose()

    for node in list(root.find_all(True)):
        if not _is_alive(node):
            continue
        if node.name in _DROP_TAGS:
            node.decompose()
            continue
        text = _norm(node.get_text(" ", strip=True)).lower()
        if text and text in _JUNK_TEXTS:
            node.decompose()


def extract_main_root(soup: BeautifulSoup) -> Tag:
    for selector in config.LOADER_CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node is None or not _is_alive(node):
            continue
        chars = _text_len(node)
        if chars >= config.LOADER_MIN_ROOT_CHARS:
            log.info("content_root", selector=selector, chars=chars)
            return node

    fallback = _largest_text_container(soup)
    if fallback is None:
        raise StructureError("no content root matched and no text container found")
    log.warn(
        "content_root_fallback",
        reason="no pinned selector matched",
        chars=_text_len(fallback),
    )
    return fallback


def _largest_text_container(soup: BeautifulSoup) -> Tag | None:
    best: Tag | None = None
    best_chars = 0
    for node in soup.find_all(("main", "article", "section", "div", "td", "body")):
        if not _is_alive(node):
            continue
        chars = _text_len(node)
        if chars > best_chars:
            best, best_chars = node, chars
    return best


def _cell_text(cell: Tag) -> str:
    return _norm(cell.get_text(" ", strip=True)).replace("|", "\\|")


def _table_rows(table: Tag) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        cells = tr.find_all(["th", "td"], recursive=False) or tr.find_all(["th", "td"])
        values = [_cell_text(cell) for cell in cells]
        if any(values):
            rows.append(values)
    return rows


def table_to_markdown(table: Tag) -> Block:
    rows = _table_rows(table)
    if not rows:
        raise StructureError("table has no rows with text")

    thead = table.find("thead")
    head_row = _table_rows(thead) if thead is not None else []
    if head_row:
        header = head_row[0]
        body = rows[1:]
    else:
        header = rows[0]
        body = rows[1:]

    width = max([len(header)] + [len(row) for row in body])
    header = header + [""] * (width - len(header))
    body = [row + [""] * (width - len(row)) for row in body] or [[] * width]
    body = [row if any(row) else [] for row in body]

    lines = [_md_row(header), _md_row(["---"] * width)]
    lines.extend(_md_row(row) for row in body)
    return Block(kind="table", text="\n".join(lines))


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _kv_pairs(node: Tag) -> list[tuple[str, str]] | None:
    name = getattr(node, "name", None)
    if name in _STRUCTURAL_TAGS:
        return None
    if name in _HEADING_TAGS:
        return None
    if node.find(list(_HEADING_TAGS)) is not None:
        return None
    if name == "dl":
        return _dl_pairs(node)
    if name in ("ul", "ol"):
        return _li_pairs(node)

    children = _child_elements(node)
    if len(children) < 2:
        return None

    if len(children) == 2 and all(_is_terminal(child) for child in children):
        pair = _make_pair(children[0], children[1])
        if pair is not None:
            return [pair]

    nested = [_kv_pairs(child) for child in children]
    if all(pairs for pairs in nested):
        return [pair for pairs in nested for pair in pairs]
    return None


def _child_elements(node: Tag) -> list[Tag]:
    return [
        child
        for child in node.find_all(True, recursive=False)
        if _is_alive(child) and child.name not in _DROP_TAGS
    ]


def _is_terminal(node: Tag) -> bool:
    return len(_child_elements(node)) < 2


def _make_pair(label_node: Tag, value_node: Tag) -> tuple[str, str] | None:
    if _is_interactive(label_node) and _is_interactive(value_node):
        return None
    label = _norm(label_node.get_text(" ", strip=True))
    value = _norm(value_node.get_text(" ", strip=True))
    if not label or not value or len(label) > _KV_MAX_LABEL_CHARS:
        return None
    if ":" not in label and (
        _KV_REJECT_LABEL.search(label) or _KV_REJECT_NUMERIC_LABEL.match(label)
    ):
        return None
    return label, value


def _is_interactive(node: Tag) -> bool:
    classes = node.get("class") or ()
    return any(name in classes for name in _INTERACTIVE_CLASSES)


def _dl_pairs(node: Tag) -> list[tuple[str, str]] | None:
    pairs: list[tuple[str, str]] = []
    label = ""
    for child in node.find_all(["dt", "dd"], recursive=False):
        text = _norm(child.get_text(" ", strip=True))
        if not text:
            continue
        if child.name == "dt":
            label = text
        elif label:
            pairs.append((label, text))
            label = ""
    return pairs or None


def _li_pairs(node: Tag) -> list[tuple[str, str]] | None:
    pairs: list[tuple[str, str]] = []
    for item in node.find_all("li", recursive=False):
        text = _norm(item.get_text(" ", strip=True))
        if not text:
            continue
        label, separator, value = text.partition(":")
        if not separator or not value.strip() or len(label) > _KV_MAX_LABEL_CHARS:
            return None
        pairs.append((label.strip(), value.strip()))
    return pairs or None


def keyvalue_to_block(node: Tag) -> Block:
    pairs = _kv_pairs(node)
    if not pairs:
        raise StructureError(f"<{node.name}> is not a label/value group")
    return Block(kind="keyvalue", text="\n".join(_kv_line(p) for p in pairs))


def _kv_line(pair: tuple[str, str]) -> str:
    label, value = pair
    separator = " " if ":" in label else ": "
    return f"{label}{separator}{value}"


def _list_block(node: Tag) -> Block:
    pairs = _kv_pairs(node)
    if pairs:
        return Block(kind="keyvalue", text="\n".join(_kv_line(p) for p in pairs))
    items = [
        _norm(item.get_text(" ", strip=True))
        for item in node.find_all("li", recursive=False)
    ]
    items = [item for item in items if item]
    if not items:
        raise StructureError(f"<{node.name}> has no list items with text")
    return Block(kind="list", text="\n".join(f"- {item}" for item in items))


def _document_title(soup: BeautifulSoup, fallback: str) -> str:
    heading = soup.find("h1")
    if heading is not None:
        text = _norm(heading.get_text(" ", strip=True))
        if text:
            return text
    if soup.title is not None:
        text = _norm(soup.title.get_text(" ", strip=True))
        if text:
            return text
    return fallback


def _tag_pills(root: Tag) -> list[str]:
    pills: list[str] = []
    for container in root.select("[class*=pills_container]"):
        for tag in container.find_all(["a", "span", "div"]):
            text = _norm(tag.get_text(" ", strip=True))
            if text and text not in pills:
                pills.append(text)
    return pills


def _pill_blocks(pills: list[str]) -> list[Block]:
    blocks: list[Block] = []
    for pill in pills:
        if _RISK_PILL.search(pill):
            blocks.append(
                Block(kind="keyvalue", text=f"{config.LOADER_RISK_PILL_LABEL}: {pill}")
            )
        else:
            blocks.append(Block(kind="text", text=pill))
    return blocks


def _overview_section(sections: list[Section]) -> Section | None:
    for section in sections:
        if section.heading == _OVERVIEW_HEADING:
            return section
    return None


def _ensure_overview(sections: list[Section]) -> Section:
    overview = _overview_section(sections)
    if overview is not None:
        return overview
    overview = Section(heading=_OVERVIEW_HEADING, level=0, heading_path=[_OVERVIEW_HEADING])
    sections.insert(0, overview)
    return overview


def _drop_empty_sections(sections: list[Section]) -> list[Section]:
    kept: list[Section] = []
    for section in sections:
        section.blocks = [block for block in section.blocks if block.text.strip()]
        if section.heading == _OVERVIEW_HEADING or section.blocks:
            kept.append(section)
    return kept


def _extract_structure(root: Tag) -> list[Section]:
    sections: list[Section] = []
    stack: list[str] = []
    buffer: list[str] = []
    state: dict[str, Section | None] = {"current": None}

    def emit(block: Block) -> None:
        current = state["current"]
        if current is None:
            current = _ensure_overview(sections)
            state["current"] = current
        current.blocks.append(block)

    def flush() -> None:
        text = _norm(" ".join(buffer))
        buffer.clear()
        if text:
            emit(Block(kind="text", text=text))

    def start_section(heading: str, level: int) -> None:
        flush()
        if _HEADING_PATH_MODE == "tree":
            del stack[level - 1 :]
            stack.append(heading)
        else:
            stack.clear()
            stack.append(heading)
        section = Section(heading=heading, level=level, heading_path=list(stack))
        sections.append(section)
        state["current"] = section

    def walk(node: Tag) -> None:
        for child in node.children:
            name = getattr(child, "name", None)
            if name is None:
                if type(child) is NavigableString:
                    text = _norm(str(child))
                    if text:
                        buffer.append(text)
                continue
            if not _is_alive(child) or name in _DROP_TAGS:
                continue
            if name in _HEADING_TAGS and _HEADING_LEVEL[name] <= _MAX_HEADING_LEVEL:
                if child.find(list(_HEADING_TAGS)) is not None:
                    walk(child)
                    continue
                start_section(_norm(child.get_text(" ", strip=True)), _HEADING_LEVEL[name])
                continue
            if name == "table":
                flush()
                emit(table_to_markdown(child))
                continue
            if name in _LIST_TAGS:
                flush()
                emit(_list_block(child))
                continue
            if _kv_pairs(child) is not None:
                flush()
                emit(keyvalue_to_block(child))
                continue
            if child.find(True) is None:
                text = _norm(child.get_text(" ", strip=True))
                if text:
                    buffer.append(text)
                continue
            walk(child)

    walk(root)
    flush()
    return _drop_empty_sections(sections)


def parse_document(fetch_result: FetchResult) -> Document:
    path = config.BASE_DIR / fetch_result.html_path
    if not path.is_absolute():
        path = config.BASE_DIR / path
    markup = path.read_text(encoding="utf-8", errors="replace")

    soup = _soup(markup)
    title = _document_title(soup, fetch_result.scheme_name)
    root = extract_main_root(soup)
    pills = _tag_pills(root)

    cleaned = _soup(clean_html(str(root)))
    sections = _extract_structure(cleaned)

    if pills:
        overview = _ensure_overview(sections)
        overview.blocks = _pill_blocks(pills) + overview.blocks

    return Document(
        doc_id=hashlib.sha1(fetch_result.url.encode("utf-8")).hexdigest()[:12],
        title=title,
        scheme=fetch_result.scheme,
        scheme_name=fetch_result.scheme_name,
        category=fetch_result.category,
        source_url=fetch_result.url,
        fetched_at=fetch_result.fetched_at,
        sections=sections,
    )


def _doc_chars(document: Document) -> int:
    return sum(len(block.text) for section in document.sections for block in section.blocks)


def _fetch_results() -> list[FetchResult]:
    if config.FETCH_MANIFEST.is_file():
        try:
            payload = json.loads(config.FETCH_MANIFEST.read_text(encoding="utf-8"))
            results = [FetchResult(**entry) for entry in payload.get("sources", [])]
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            raise StructureError(f"fetch manifest unreadable: {exc}") from exc
        if results and all((config.BASE_DIR / r.html_path).is_file() for r in results):
            return results
    return fetch_all(use_cache=True)


def dump_documents(documents: list[Document] | None = None) -> Path:
    docs = documents if documents is not None else load_documents()
    config.ensure_dirs()
    config.RAW_DOCS_JSON.write_text(
        json.dumps([asdict(doc) for doc in docs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(
        "raw_docs",
        path=str(config.RAW_DOCS_JSON.relative_to(config.BASE_DIR)),
        n=len(docs),
    )
    return config.RAW_DOCS_JSON


def load_documents() -> list[Document]:
    documents = []
    for result in _fetch_results():
        document = parse_document(result)
        log.info(
            "parse",
            scheme=document.scheme,
            sections=len(document.sections),
            blocks=sum(len(s.blocks) for s in document.sections),
            chars=_doc_chars(document),
        )
        documents.append(document)
    return documents


def _print_parse_summary(documents: list[Document]) -> None:
    print()
    header = (
        f"{'scheme':<24} {'sections':>8} {'tables':>7} {'keyvalues':>10} "
        f"{'lists':>6} {'texts':>6} {'chars':>8}  title"
    )
    print(header)
    print("-" * len(header))
    for document in documents:
        kinds = [block.kind for section in document.sections for block in section.blocks]
        print(
            f"{document.scheme:<24} {len(document.sections):>8} "
            f"{kinds.count('table'):>7} {kinds.count('keyvalue'):>10} "
            f"{kinds.count('list'):>6} {kinds.count('text'):>6} "
            f"{_doc_chars(document):>8,}  {document.title}"
        )
    total_sections = sum(len(d.sections) for d in documents)
    total_chars = sum(_doc_chars(d) for d in documents)
    print("-" * len(header))
    print(f"{'TOTAL':<24} {total_sections:>8} {'':>7} {'':>10} {'':>6} {'':>6} {total_chars:>8,}")


def _print_fetch_summary(results: list[FetchResult]) -> None:
    print()
    header = f"{'scheme':<24} {'status':>6} {'bytes':>10} {'attempts':>8} {'cached':>7}  fetched_at"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.scheme:<24} {r.status:>6} {r.n_bytes:>10,} {r.attempts:>8} "
            f"{str(r.cached).lower():>7}  {r.fetched_at}"
        )
    total = sum(r.n_bytes for r in results)
    print("-" * len(header))
    print(f"{'TOTAL':<24} {len(results):>6} {total:>10,}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the allowlisted HDFC scheme pages and extract their structure."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore cached raw HTML and hit the network for every source",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="clean cached raw HTML into heading-tree Documents and write raw_docs.json",
    )
    args = parser.parse_args()

    try:
        if args.parse:
            documents = load_documents()
            dump_documents(documents)
            _print_parse_summary(documents)
            return 0

        results = fetch_all(use_cache=not args.no_cache)
    except (SourceValidationError, FetchError, FileNotFoundError, StructureError) as exc:
        log.error("aborted", reason=str(exc))
        return 1

    _print_fetch_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
