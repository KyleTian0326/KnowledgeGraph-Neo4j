import argparse
import shutil
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

from build_kg_with_deepseek import build as build_kg
from build_kg_with_deepseek import read_text_file
from build_vector_chunks import build as build_vector_chunks
from ocr_pdf_to_text import ocr_pdf


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def parse_pages(value: str | None) -> str:
    if not value:
        return "all"
    value = value.strip()
    return value or "all"


def safe_stem(path: Path) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in path.stem)
    return cleaned.strip("_") or "document"


def has_pdf_text(path: Path, sample_pages: int = 5, min_chars: int = 80) -> bool:
    reader = PdfReader(str(path))
    total_chars = 0
    for page in reader.pages[:sample_pages]:
        total_chars += len((page.extract_text() or "").strip())
        if total_chars >= min_chars:
            return True
    return False


def extract_pdf_text_pages(path: Path, output_dir: Path, pages: str) -> list[Path]:
    reader = PdfReader(str(path))
    selected_pages = select_pages(pages, len(reader.pages))
    outputs: list[Path] = []
    for page_number in selected_pages:
        page = reader.pages[page_number - 1]
        text = page.extract_text() or ""
        output_path = output_dir / f"{safe_stem(path)}_page_{page_number:04d}.txt"
        output_path.write_text(f"===== PAGE {page_number} =====\n{text.strip()}\n", encoding="utf-8")
        outputs.append(output_path)
    return outputs


def select_pages(value: str, total_pages: int) -> list[int]:
    if value.lower() in {"all", "*"}:
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

    files = [
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def prepare_text_files(
    source_dir: Path,
    data_dir: Path,
    filename: str | None,
    pages: str,
    force_ocr: bool,
    zoom: float,
) -> list[Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    source_files = collect_source_files(source_dir, filename)
    if not source_files:
        raise FileNotFoundError(f"No supported files found in {source_dir}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = data_dir / f"ingested_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_files: list[Path] = []
    selected_pages = parse_pages(pages)

    for source_path in source_files:
        suffix = source_path.suffix.lower()
        print(f"\nPreparing: {source_path}")

        if suffix in {".txt", ".md"}:
            output_path = output_dir / f"{safe_stem(source_path)}.txt"
            text = read_text_file(source_path)
            output_path.write_text(text.strip() + "\n", encoding="utf-8")
            prepared_files.append(output_path)
            print(f"- copied text to {output_path}")
            continue

        if suffix == ".pdf":
            if force_ocr or not has_pdf_text(source_path):
                print(f"- OCR PDF pages: {selected_pages}")
                page_numbers = select_pages(selected_pages, len(PdfReader(str(source_path)).pages))
                for page_number in page_numbers:
                    output_path = output_dir / f"{safe_stem(source_path)}_page_{page_number:04d}.txt"
                    ocr_pdf(source_path, output_path, str(page_number), zoom=zoom, write_page_files=False)
                    prepared_files.append(output_path)
            else:
                print(f"- extract embedded PDF text pages: {selected_pages}")
                prepared_files.extend(extract_pdf_text_pages(source_path, output_dir, selected_pages))

    return prepared_files


def copy_latest_json(output_path: Path, latest_path: Path) -> None:
    if output_path.exists():
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, latest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally ingest files from source/, build KG, and update vector chunks."
    )
    parser.add_argument("--source", default="source", help="Folder containing new source files.")
    parser.add_argument("--data", default="data", help="Folder for extracted text outputs.")
    parser.add_argument("--file", default=None, help="Only process one file name inside source/.")
    parser.add_argument("--pages", default=None, help="PDF pages, e.g. 30-40, 1,5,8, or all. Defaults to all.")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR for PDFs even if embedded text exists.")
    parser.add_argument("--zoom", type=float, default=1.8, help="PDF render zoom for OCR.")
    parser.add_argument("--dry-run", action="store_true", help="Only extract KG JSON, do not write Neo4j.")
    parser.add_argument("--skip-kg", action="store_true", help="Only prepare text and vector chunks, skip KG extraction.")
    parser.add_argument("--skip-vector", action="store_true", help="Only prepare text and KG extraction, skip vector chunks.")
    parser.add_argument("--reset-vector", action="store_true", help="Delete old Chunk nodes before writing new chunks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    data_dir = Path(args.data)

    prepared_files = prepare_text_files(
        source_dir=source_dir,
        data_dir=data_dir,
        filename=args.file,
        pages=args.pages,
        force_ocr=args.force_ocr,
        zoom=args.zoom,
    )

    if not prepared_files:
        raise RuntimeError("No text files were prepared.")

    prepared_dir = prepared_files[0].parent
    extraction_output = Path("output") / f"{prepared_dir.name}_kg_extraction.json"

    if not args.skip_kg:
        print("\n=== Build knowledge graph with DeepSeek ===")
        build_kg(
            prepared_dir,
            dry_run=args.dry_run,
            output_path=extraction_output,
        )
        copy_latest_json(extraction_output, Path("output/deepseek_kg_extraction.json"))

    if not args.skip_vector and not args.dry_run:
        print("\n=== Build vector chunks and index ===")
        build_vector_chunks(prepared_dir, reset=args.reset_vector)

    print("\nIngestion finished.")
    print(f"Prepared text folder: {prepared_dir}")
    if not args.skip_kg:
        print(f"Extraction JSON: {extraction_output}")
    if args.dry_run:
        print("Dry-run mode: Neo4j was not updated.")


if __name__ == "__main__":
    main()
