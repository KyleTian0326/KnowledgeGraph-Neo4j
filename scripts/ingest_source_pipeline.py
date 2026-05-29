import argparse
import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index
from openai import OpenAI
from rapidocr_onnxruntime import RapidOCR

from build_kg_with_deepseek import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    create_constraints,
    extract_chunk,
    read_text_file,
    split_text,
    write_extraction,
    write_output_json,
)
from build_vector_chunks import (
    CHUNK_EMBEDDING_PROPERTY,
    CHUNK_LABEL,
    EMBEDDING_DIMENSIONS,
    VECTOR_INDEX_NAME,
    create_chunk_constraints,
    write_chunk,
)
from local_embeddings import build_embedder
from source_metadata import page_label
from progress import ProgressTracker
from page_windowing import PageText, page_section, safe_stem, windowed_consecutive_pages, write_page_window


load_dotenv()

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
SENTINEL = object()


@dataclass
class PreparedDocument:
    path: Path
    source_name: str


class ThreadSafeJsonLog:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.lock = threading.Lock()
        self.items = self._load_existing()
        self.processed = {
            (str(item.get("source")), int(item.get("chunk")))
            for item in self.items
            if isinstance(item, dict) and item.get("source") and item.get("chunk")
        }

    def _load_existing(self) -> list[dict[str, Any]]:
        if not self.output_path.exists():
            return []
        try:
            data = json.loads(self.output_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def already_done(self, source: str, chunk_index: int) -> bool:
        with self.lock:
            return (source, chunk_index) in self.processed

    def append(self, item: dict[str, Any]) -> None:
        with self.lock:
            self.items.append(item)
            if item.get("source") and item.get("chunk"):
                self.processed.add((str(item["source"]), int(item["chunk"])))
            write_output_json(self.items, self.output_path)


def parse_pages(value: str | None, total_pages: int) -> list[int]:
    if not value or value.lower() in {"all", "*"}:
        return list(range(1, total_pages + 1))

    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            pages.update(range(int(start), int(end) + 1))
        else:
            pages.add(int(part))
    return [page for page in sorted(pages) if 1 <= page <= total_pages]


def collect_source_files(source_dir: Path, filename: str | None) -> list[Path]:
    if filename:
        file_path = source_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        return [file_path]

    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def count_work_items(source_files: list[Path], pages: str | None, page_window: int, page_window_overlap: int) -> int:
    total = 0
    for source_path in source_files:
        suffix = source_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            total += 1
        elif suffix == ".pdf":
            doc = fitz.open(source_path)
            page_numbers = parse_pages(pages, doc.page_count)
            if page_window > 0:
                total += len(
                    windowed_consecutive_pages(
                        [PageText(page, "") for page in page_numbers],
                        page_window,
                        page_window_overlap,
                    )
                )
            else:
                total += len(page_numbers)
            doc.close()
    return total


def pdf_has_text(path: Path, sample_pages: int = 5, min_chars: int = 80) -> bool:
    doc = fitz.open(path)
    total_chars = 0
    for page_index in range(min(sample_pages, doc.page_count)):
        total_chars += len(doc.load_page(page_index).get_text().strip())
        if total_chars >= min_chars:
            doc.close()
            return True
    doc.close()
    return False


def write_text_page(path: Path, page_number: int, text: str, output_dir: Path) -> PreparedDocument:
    output_path = output_dir / f"{safe_stem(path)}_page_{page_number:04d}.txt"
    output_path.write_text(page_section(page_number, text), encoding="utf-8")
    return PreparedDocument(path=output_path, source_name=output_path.name)


def enqueue_page_window(source_path: Path, pages: list[PageText], output_dir: Path, work_queue: queue.Queue) -> None:
    output_path = write_page_window(source_path, pages, output_dir)
    work_queue.put(PreparedDocument(path=output_path, source_name=output_path.name))


def producer(
    source_files: list[Path],
    output_dir: Path,
    work_queue: queue.Queue,
    pages: str | None,
    force_ocr: bool,
    zoom: float,
    page_window: int,
    page_window_overlap: int,
    progress: ProgressTracker,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_files:
        suffix = source_path.suffix.lower()
        print(f"\nPreparing: {source_path}")

        if suffix in {".txt", ".md"}:
            output_path = output_dir / f"{safe_stem(source_path)}.txt"
            progress.render(current=f"prepare {source_path.name}", force=True)
            output_path.write_text(read_text_file(source_path).strip() + "\n", encoding="utf-8")
            work_queue.put(PreparedDocument(path=output_path, source_name=output_path.name))
            continue

        if suffix != ".pdf":
            continue

        doc = fitz.open(source_path)
        selected_pages = parse_pages(pages, doc.page_count)
        use_ocr = force_ocr or not pdf_has_text(source_path)
        ocr = RapidOCR() if use_ocr else None
        mode = "OCR" if use_ocr else "text"
        print(f"- PDF mode: {mode}, pages: {len(selected_pages)}")
        if page_window > 0:
            print(f"- page window merge: {page_window} pages, overlap {page_window_overlap}")
            buffered_pages: list[PageText] = []

        for page_number in selected_pages:
            page = doc.load_page(page_number - 1)
            progress.render(current=f"{mode} {source_path.name} page {page_number}/{doc.page_count}", force=True)
            if use_ocr:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                image_path = output_dir / f"{safe_stem(source_path)}_page_{page_number:04d}.png"
                pix.save(image_path)
                result, _ = ocr(str(image_path))
                lines = [line[1] for line in result] if result else []
                text = "\n".join(lines)
            else:
                text = page.get_text()

            if page_window > 0:
                if buffered_pages and page_number != buffered_pages[-1].page + 1:
                    if buffered_pages:
                        enqueue_page_window(source_path, buffered_pages, output_dir, work_queue)
                    buffered_pages = []
                buffered_pages.append(PageText(page_number, text))
                if len(buffered_pages) >= page_window:
                    window = buffered_pages[:page_window]
                    enqueue_page_window(source_path, window, output_dir, work_queue)
                    keep = page_window_overlap
                    buffered_pages = buffered_pages[-keep:] if keep else []
            else:
                prepared = write_text_page(source_path, page_number, text, output_dir)
                work_queue.put(prepared)

        if page_window > 0 and buffered_pages:
            enqueue_page_window(source_path, buffered_pages, output_dir, work_queue)
        doc.close()


def consumer(
    worker_id: int,
    work_queue: queue.Queue,
    driver,
    json_log: ThreadSafeJsonLog,
    vector_lock: threading.Lock,
    skip_kg: bool,
    skip_vector: bool,
    progress: ProgressTracker,
) -> None:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    embedder = build_embedder()

    while True:
        item = work_queue.get()
        try:
            if item is SENTINEL:
                return

            assert isinstance(item, PreparedDocument)
            text = read_text_file(item.path)
            chunks = split_text(text, source=item.source_name)
            progress.render(current=f"worker {worker_id} {item.source_name}, chunks={len(chunks)}", force=True)

            for chunk in chunks:
                if json_log.already_done(chunk.source, chunk.index):
                    progress.render(current=f"worker {worker_id} skip {chunk.source} chunk {chunk.index}", force=True)
                    continue

                extraction = {"nodes": [], "relationships": []}
                if not skip_kg:
                    extraction = extract_chunk(client, chunk)
                    write_extraction(driver, extraction, source=chunk.source, chunk=chunk)

                if not skip_vector:
                    embedding = embedder.embed_query(chunk.text)
                    with vector_lock:
                        write_chunk(driver, chunk, embedding)

                json_log.append(
                    {
                        "source": chunk.source,
                        "chunk": chunk.index,
                        "document": chunk.document,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "page": page_label(chunk.page_start, chunk.page_end),
                        "extraction": extraction,
                    }
                )
            progress.advance(current=f"worker {worker_id} finished {item.source_name}")
        finally:
            work_queue.task_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline ingest: OCR/extract pages while DeepSeek workers build KG and vector chunks."
    )
    parser.add_argument("--source", default="source", help="Folder containing new source files.")
    parser.add_argument("--file", default=None, help="Only process one file name inside source/.")
    parser.add_argument("--pages", default=None, help="PDF pages, e.g. 30-40, 1,5,8, or all. Defaults to all.")
    parser.add_argument("--data", default="data", help="Folder for prepared page text files.")
    parser.add_argument("--output", default=None, help="Extraction JSON path. Defaults to output/pipeline_<time>_kg.json.")
    parser.add_argument("--workers", type=int, default=3, help="DeepSeek extraction worker count.")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR for PDFs even if embedded text exists.")
    parser.add_argument("--zoom", type=float, default=1.8, help="PDF render zoom for OCR.")
    parser.add_argument("--page-window", type=int, default=20, help="Merge this many consecutive PDF pages before chunking. Use 0 to keep one file per page.")
    parser.add_argument("--page-window-overlap", type=int, default=1, help="How many pages adjacent merged windows overlap.")
    parser.add_argument("--skip-kg", action="store_true", help="Only prepare text and vector chunks, skip KG extraction.")
    parser.add_argument("--skip-vector", action="store_true", help="Only build KG, skip vector chunks.")
    parser.add_argument("--reset-vector", action="store_true", help="Delete old Chunk nodes before writing new chunks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-your-"):
        raise RuntimeError("Please set DEEPSEEK_API_KEY in .env")

    source_dir = Path(args.source)
    source_files = collect_source_files(source_dir, args.file)
    if not source_files:
        raise FileNotFoundError(f"No supported files found in {source_dir}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    prepared_dir = Path(args.data) / f"pipeline_{run_id}"
    output_path = Path(args.output) if args.output else Path("output") / f"pipeline_{run_id}_kg_extraction.json"
    json_log = ThreadSafeJsonLog(output_path)
    if args.page_window > 0 and args.page_window_overlap >= args.page_window:
        raise ValueError("--page-window-overlap must be smaller than --page-window")

    total_work = count_work_items(source_files, args.pages, args.page_window, args.page_window_overlap)
    progress = ProgressTracker(total_work, label="Pipeline")
    progress.render(current="ready", force=True)

    work_queue: queue.Queue = queue.Queue(maxsize=max(2, args.workers * 2))
    vector_lock = threading.Lock()

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        driver.verify_connectivity()
        if not args.skip_kg:
            create_constraints(driver)
        if not args.skip_vector:
            create_chunk_constraints(driver)
            if args.reset_vector:
                driver.execute_query(
                    f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS",
                    database_=NEO4J_DATABASE,
                )
                driver.execute_query(
                    f"MATCH (c:{CHUNK_LABEL}) WHERE c.vector_kg = true DETACH DELETE c",
                    database_=NEO4J_DATABASE,
                )

        threads = [
            threading.Thread(
                target=consumer,
                args=(
                    idx + 1,
                    work_queue,
                    driver,
                    json_log,
                    vector_lock,
                    args.skip_kg,
                    args.skip_vector,
                    progress,
                ),
                daemon=True,
            )
            for idx in range(max(1, args.workers))
        ]
        for thread in threads:
            thread.start()

        producer(
            source_files=source_files,
            output_dir=prepared_dir,
            work_queue=work_queue,
            pages=args.pages,
            force_ocr=args.force_ocr,
            zoom=args.zoom,
            page_window=args.page_window,
            page_window_overlap=args.page_window_overlap,
            progress=progress,
        )

        for _ in threads:
            work_queue.put(SENTINEL)

        work_queue.join()
        for thread in threads:
            thread.join(timeout=1)

        if not args.skip_vector:
            create_vector_index(
                driver,
                VECTOR_INDEX_NAME,
                label=CHUNK_LABEL,
                embedding_property=CHUNK_EMBEDDING_PROPERTY,
                dimensions=EMBEDDING_DIMENSIONS,
                similarity_fn="cosine",
                fail_if_exists=False,
                neo4j_database=NEO4J_DATABASE,
            )

    progress.finish()

    print("\nPipeline ingestion finished.")
    print(f"Prepared text folder: {prepared_dir}")
    print(f"Extraction JSON: {output_path}")
    print(f"DeepSeek workers: {max(1, args.workers)}")


if __name__ == "__main__":
    main()
