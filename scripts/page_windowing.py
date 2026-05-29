from dataclasses import dataclass
from pathlib import Path
import re

from source_metadata import strip_page_markers


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


def safe_stem(path: Path) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in path.stem)
    return cleaned.strip("_") or "document"


def document_stem(path: Path) -> str:
    stem = safe_stem(path)
    stem = re.sub(r"_page_\d{1,6}$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_pages_\d{1,6}_\d{1,6}$", "", stem, flags=re.IGNORECASE)
    return stem.strip("_") or "document"


def page_section(page_number: int, text: str) -> str:
    return f"===== PAGE {page_number} =====\n{strip_page_markers(text).strip()}\n"


def windowed_pages(pages: list[PageText], window_size: int, overlap: int) -> list[list[PageText]]:
    if window_size <= 0:
        return [pages] if pages else []
    if overlap < 0:
        raise ValueError("page window overlap must be >= 0")
    if overlap >= window_size:
        raise ValueError("page window overlap must be smaller than page window size")

    windows: list[list[PageText]] = []
    step = window_size - overlap
    start = 0
    while start < len(pages):
        window = pages[start : start + window_size]
        if window:
            windows.append(window)
        if start + window_size >= len(pages):
            break
        start += step
    return windows


def consecutive_page_groups(pages: list[PageText]) -> list[list[PageText]]:
    groups: list[list[PageText]] = []
    current: list[PageText] = []
    for page in pages:
        if current and page.page != current[-1].page + 1:
            groups.append(current)
            current = []
        current.append(page)
    if current:
        groups.append(current)
    return groups


def windowed_consecutive_pages(pages: list[PageText], window_size: int, overlap: int) -> list[list[PageText]]:
    windows: list[list[PageText]] = []
    for group in consecutive_page_groups(pages):
        windows.extend(windowed_pages(group, window_size, overlap))
    return windows


def window_filename(source_path: Path, pages: list[PageText]) -> str:
    stem = document_stem(source_path)
    if not pages:
        return f"{stem}.txt"
    first = pages[0].page
    last = pages[-1].page
    if first == last:
        return f"{stem}_page_{first:04d}.txt"
    return f"{stem}_pages_{first:04d}_{last:04d}.txt"


def write_page_window(source_path: Path, pages: list[PageText], output_dir: Path) -> Path:
    output_path = output_dir / window_filename(source_path, pages)
    output_path.write_text("\n".join(page_section(item.page, item.text).strip() for item in pages) + "\n", encoding="utf-8")
    return output_path
