import argparse
import json
import re
from pathlib import Path
from typing import Any

from build_kg_with_deepseek import iter_input_files, read_document, split_text
from source_metadata import (
    PAGE_FILE_RE,
    clean_document_name,
    find_page_spans,
    page_from_source,
    page_neighbor_ref,
    strip_page_markers,
)


TRAILING_PUNCT_RE = re.compile(r"[。！？.!?;；:：”’）)\]】》」』]\s*$")


def compact_text(text: str) -> str:
    text = strip_page_markers(text)
    text = re.sub(r"\s+", "", text)
    return text


def text_tail(text: str, width: int) -> str:
    compact = compact_text(text)
    return compact[-width:]


def text_head(text: str, width: int) -> str:
    compact = compact_text(text)
    return compact[:width]


def chunk_covers_pages(chunk: Any, page_a: int, page_b: int) -> bool:
    start = chunk.page_start
    end = chunk.page_end if chunk.page_end is not None else start
    if start is None or end is None:
        return False
    return int(start) <= page_a and int(end) >= page_b


def chunk_contains_boundary_text(chunk: Any, tail: str, head: str, min_chars: int) -> bool:
    compact = compact_text(chunk.text)
    return bool(tail[:min_chars] and head[:min_chars] and tail[-min_chars:] in compact and head[:min_chars] in compact)


def page_segments(text: str) -> list[dict[str, Any]]:
    spans = find_page_spans(text)
    pages = []
    for index, span in enumerate(spans):
        start = span.end
        end = spans[index + 1].start if index + 1 < len(spans) else len(text)
        pages.append({"page": span.page, "text": text[start:end]})
    return pages


def score_boundary(
    chunks: list[Any],
    page_a: int,
    page_b: int,
    tail: str,
    head: str,
    min_chars: int,
    end_text: str,
) -> dict[str, Any]:
    crossing = [chunk for chunk in chunks if chunk_covers_pages(chunk, page_a, page_b)]
    bridged = [chunk for chunk in crossing if chunk_contains_boundary_text(chunk, tail, head, min_chars)]
    likely_sentence_split = not bool(TRAILING_PUNCT_RE.search(end_text.strip()))
    status = "ok" if bridged else ("warning" if crossing else "fail")
    return {
        "from_page": page_a,
        "to_page": page_b,
        "boundary_ref": page_neighbor_ref("", page_a, "to_next"),
        "likely_sentence_split": likely_sentence_split,
        "cross_page_chunks": [chunk.index for chunk in crossing],
        "bridging_chunks": [chunk.index for chunk in bridged],
        "tail": tail,
        "head": head,
        "status": status,
    }


def audit_file(path: Path, tail_chars: int, head_chars: int, min_match_chars: int) -> dict[str, Any]:
    text = read_document(path)
    pages = page_segments(text)
    chunks = split_text(text, source=path.name)
    document = clean_document_name(path.name)
    boundaries = []
    for left, right in zip(pages, pages[1:]):
        page_a = int(left["page"])
        page_b = int(right["page"])
        if page_b != page_a + 1:
            continue
        tail = text_tail(left["text"], tail_chars)
        head = text_head(right["text"], head_chars)
        result = score_boundary(
            chunks=chunks,
            page_a=page_a,
            page_b=page_b,
            tail=tail,
            head=head,
            min_chars=min_match_chars,
            end_text=str(left["text"]),
        )
        result["boundary_ref"] = page_neighbor_ref(document, page_a, "to_next")
        boundaries.append(result)
    counts = {
        "ok": sum(1 for item in boundaries if item["status"] == "ok"),
        "warning": sum(1 for item in boundaries if item["status"] == "warning"),
        "fail": sum(1 for item in boundaries if item["status"] == "fail"),
        "likely_sentence_split": sum(1 for item in boundaries if item["likely_sentence_split"]),
    }
    return {
        "source": path.name,
        "document": document,
        "pages": len(pages),
        "chunks": len(chunks),
        "boundaries": boundaries,
        "summary": counts,
    }


def group_page_files(files: list[Path]) -> tuple[list[Path], dict[str, list[Path]]]:
    grouped: dict[str, list[Path]] = {}
    standalone = []
    for path in files:
        match = PAGE_FILE_RE.match(path.name)
        if not match:
            standalone.append(path)
            continue
        grouped.setdefault(match.group("document"), []).append(path)
    for paths in grouped.values():
        paths.sort(key=lambda item: page_from_source(item.name) or 0)
    return standalone, grouped


def audit_page_file_group(
    document_key: str,
    paths: list[Path],
    tail_chars: int,
    head_chars: int,
    min_match_chars: int,
) -> dict[str, Any]:
    document = clean_document_name(document_key)
    joined_parts = []
    page_texts: list[dict[str, Any]] = []
    for path in paths:
        page = page_from_source(path.name)
        if page is None:
            continue
        text = read_document(path)
        page_texts.append({"page": page, "text": text})
        joined_parts.append(text if find_page_spans(text) else f"===== PAGE {page} =====\n{text}")

    joined_text = "\n\n".join(joined_parts)
    chunks = split_text(joined_text, source=f"{document_key}.txt")
    boundaries = []
    for left, right in zip(page_texts, page_texts[1:]):
        page_a = int(left["page"])
        page_b = int(right["page"])
        if page_b != page_a + 1:
            continue
        tail = text_tail(left["text"], tail_chars)
        head = text_head(right["text"], head_chars)
        result = score_boundary(
            chunks=chunks,
            page_a=page_a,
            page_b=page_b,
            tail=tail,
            head=head,
            min_chars=min_match_chars,
            end_text=str(left["text"]),
        )
        result["boundary_ref"] = page_neighbor_ref(document, page_a, "to_next")
        boundaries.append(result)

    counts = {
        "ok": sum(1 for item in boundaries if item["status"] == "ok"),
        "warning": sum(1 for item in boundaries if item["status"] == "warning"),
        "fail": sum(1 for item in boundaries if item["status"] == "fail"),
        "likely_sentence_split": sum(1 for item in boundaries if item["likely_sentence_split"]),
    }
    return {
        "source": document_key,
        "document": document,
        "pages": len(page_texts),
        "chunks": len(chunks),
        "boundaries": boundaries,
        "summary": counts,
    }


def summarize(files: list[dict[str, Any]]) -> dict[str, Any]:
    total = {"boundaries": 0, "ok": 0, "warning": 0, "fail": 0, "likely_sentence_split": 0}
    for item in files:
        summary = item["summary"]
        boundary_count = len(item["boundaries"])
        total["boundaries"] += boundary_count
        for key in ("ok", "warning", "fail", "likely_sentence_split"):
            total[key] += int(summary.get(key, 0))
    total["bridge_rate"] = round(total["ok"] / total["boundaries"], 6) if total["boundaries"] else None
    total["risk_rate"] = round((total["warning"] + total["fail"]) / total["boundaries"], 6) if total["boundaries"] else None
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether page-boundary text is preserved by chunking.")
    parser.add_argument("--input", required=True, help="Prepared TXT/PDF file or folder.")
    parser.add_argument("--output", default=None, help="JSON report path.")
    parser.add_argument("--tail-chars", type=int, default=80, help="Characters taken from the end of the previous page.")
    parser.add_argument("--head-chars", type=int, default=80, help="Characters taken from the beginning of the next page.")
    parser.add_argument("--min-match-chars", type=int, default=24, help="Minimum boundary text that must appear in a chunk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = iter_input_files(Path(args.input))
    if not files:
        raise FileNotFoundError(f"No supported files found: {args.input}")
    standalone, grouped = group_page_files(files)
    results = [
        audit_file(path, args.tail_chars, args.head_chars, args.min_match_chars)
        for path in standalone
    ]
    results.extend(
        audit_page_file_group(
            document_key,
            paths,
            args.tail_chars,
            args.head_chars,
            args.min_match_chars,
        )
        for document_key, paths in grouped.items()
    )
    report = {"input": args.input, "summary": summarize(results), "files": results}
    output_path = Path(args.output) if args.output else Path("output/page_continuity_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Page continuity audit finished.")
    print(f"Files: {len(results)}")
    print(f"Boundaries: {report['summary']['boundaries']}")
    print(f"Bridge rate: {report['summary']['bridge_rate']}")
    print(f"Risk rate: {report['summary']['risk_rate']}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
