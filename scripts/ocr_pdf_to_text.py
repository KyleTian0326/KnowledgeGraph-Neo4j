import argparse
from pathlib import Path

import fitz
from rapidocr_onnxruntime import RapidOCR


def parse_pages(value: str, total_pages: int) -> list[int]:
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


def ocr_pdf(
    input_path: Path,
    output_path: Path,
    pages: str,
    zoom: float = 1.8,
    write_page_files: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(input_path)
    page_numbers = parse_pages(pages, doc.page_count)
    if not page_numbers:
        raise ValueError("No valid pages selected.")

    ocr = RapidOCR()
    sections = []
    for page_number in page_numbers:
        print(f"OCR page {page_number}/{doc.page_count}")
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image_path = output_path.parent / f"{output_path.stem}_page_{page_number}.png"
        pix.save(image_path)
        result, _ = ocr(str(image_path))
        lines = [line[1] for line in result] if result else []
        text = "\n".join(lines).strip()
        sections.append(f"\n\n===== PAGE {page_number} =====\n{text}")
        if write_page_files and len(page_numbers) > 1:
            page_output = output_path.parent / f"{output_path.stem}_page_{page_number:04d}.txt"
            page_output.write_text(f"===== PAGE {page_number} =====\n{text}\n", encoding="utf-8")

    output_path.write_text("".join(sections).strip() + "\n", encoding="utf-8")
    print(f"Saved OCR text: {output_path}")
    print(f"Pages: {len(page_numbers)}")
    print(f"Characters: {len(output_path.read_text(encoding='utf-8'))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR scanned PDF pages into UTF-8 text.")
    parser.add_argument("--input", required=True, help="Input scanned PDF.")
    parser.add_argument("--output", required=True, help="Output UTF-8 text file.")
    parser.add_argument("--pages", default="1-5", help="Pages to OCR, e.g. 20-40, 20,25,30, or all.")
    parser.add_argument("--zoom", type=float, default=1.8, help="PDF render zoom for OCR.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ocr_pdf(Path(args.input), Path(args.output), args.pages, zoom=args.zoom)


if __name__ == "__main__":
    main()
