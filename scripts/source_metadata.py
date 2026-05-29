import re
from dataclasses import dataclass


PAGE_FILE_RE = re.compile(r"^(?P<document>.+?)_page_(?P<page>\d{1,6})\.txt$", re.IGNORECASE)
PAGE_RANGE_FILE_RE = re.compile(
    r"^(?P<document>.+?)_pages_(?P<start>\d{1,6})_(?P<end>\d{1,6})\.txt$",
    re.IGNORECASE,
)
PAGE_MARKER_RE = re.compile(r"(?m)^\s*(?:===== PAGE|\[Page)\s+(?P<page>\d+)\s*(?:=====\s*|\]\s*)?$")


@dataclass(frozen=True)
class PageSpan:
    page: int
    start: int
    end: int


def clean_document_name(source: str) -> str:
    name = re.sub(r"\.(txt|md|pdf)$", "", source or "", flags=re.IGNORECASE)
    match = PAGE_FILE_RE.match(source or "")
    if match:
        name = match.group("document")
    range_match = PAGE_RANGE_FILE_RE.match(source or "")
    if range_match:
        name = range_match.group("document")
    name = name.replace("_", " ")
    name = re.split(r"\b(?:z-library|1lib|z-lib)\b", name, maxsplit=1, flags=re.IGNORECASE)[0]
    name = re.sub(r"\(\s*\d+\s*\)$", "", name).strip()
    return re.sub(r"\s+", " ", name).strip() or (source or "document")


def page_from_source(source: str) -> int | None:
    match = PAGE_FILE_RE.match(source or "")
    if not match:
        return None
    return int(match.group("page"))


def page_label(page_start: int | None, page_end: int | None = None) -> str | None:
    if page_start is None:
        return None
    if page_end is None or page_end == page_start:
        return str(page_start)
    return f"{page_start}-{page_end}"


def source_ref(document: str, page_start: int | None, page_end: int | None, chunk_index: int | None) -> str:
    page = page_label(page_start, page_end) or "unknown"
    chunk = chunk_index if chunk_index is not None else "unknown"
    return f"{document}#page={page}#chunk={chunk}"


def page_neighbor_ref(document: str, page: int, side: str) -> str:
    return f"{document}#page={page}#boundary={side}"


def find_page_spans(text: str) -> list[PageSpan]:
    matches = list(PAGE_MARKER_RE.finditer(text or ""))
    spans: list[PageSpan] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans.append(PageSpan(page=int(match.group("page")), start=start, end=end))
    return spans


def strip_page_markers(text: str) -> str:
    return PAGE_MARKER_RE.sub("", text or "").strip()


def page_range_for_offsets(
    source: str,
    text: str,
    start: int,
    end: int,
    page_spans: list[PageSpan] | None = None,
) -> tuple[int | None, int | None]:
    source_page = page_from_source(source)
    if source_page is not None:
        return source_page, source_page

    spans = page_spans if page_spans is not None else find_page_spans(text)
    pages = [span.page for span in spans if span.start < end and span.end > start]
    if not pages:
        segment = text[start:end]
        marker_pages = [int(match.group("page")) for match in PAGE_MARKER_RE.finditer(segment)]
        pages = marker_pages
    if not pages:
        return None, None
    return min(pages), max(pages)
