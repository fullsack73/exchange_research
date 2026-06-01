from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def convert_pdf(pdf_path: Path, dpi: int) -> int:
    output_dir = pdf_path.with_suffix("")
    output_dir.mkdir(exist_ok=True)

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            output_path = output_dir / f"{page_number}.png"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(output_path)

        return document.page_count


def find_pdfs(target_dir: Path) -> list[Path]:
    return sorted(path for path in target_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert each PDF in a directory to per-page PNG files stored in a folder named after the PDF."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing PDF files. Defaults to the current directory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Render resolution for output PNG files. Defaults to 200.",
    )
    args = parser.parse_args()

    target_dir = Path(args.directory).expanduser().resolve()
    pdf_paths = find_pdfs(target_dir)

    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {target_dir}")

    for pdf_path in pdf_paths:
        page_count = convert_pdf(pdf_path, args.dpi)
        print(f"Converted {pdf_path.name}: {page_count} page(s) -> {pdf_path.stem}/")


if __name__ == "__main__":
    main()