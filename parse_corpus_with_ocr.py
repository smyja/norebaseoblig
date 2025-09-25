#!/usr/bin/env python3
"""Parse PDFs under regulations_corpus with an OCR fallback and page metadata."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import fitz  # type: ignore[import-not-found]
import pytesseract  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]
from pytesseract import Output  # type: ignore[import-not-found]

ROOT = Path("regulations_corpus")
OUT = Path("parsed_pages")

# Try to suppress verbose MuPDF error messages on stderr, if supported
try:  # pragma: no cover - best-effort
    fitz.TOOLS.mupdf_display_errors(False)  # type: ignore[attr-defined]
except Exception:
    pass


def load_manifest(dir_path: Path) -> Dict[str, Dict[str, Any]]:
    """Return filename keyed metadata from manifest.jsonl when present."""

    manifest: Dict[str, Dict[str, Any]] = {}
    manifest_path = dir_path / "manifest.jsonl"
    if not manifest_path.exists():
        return manifest

    for line in manifest_path.open("r", encoding="utf-8"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        filename = record.get("filename")
        if not filename:
            continue

        manifest[Path(filename).name] = record

    return manifest


def sha256_bytes(data: bytes) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def _ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    ascii_count = sum(1 for ch in s if 32 <= ord(ch) <= 126 or ch in {"\n", "\t", "\r"})
    return ascii_count / max(1, len(s))


def _ocr_image(img: Image.Image, dpi: int) -> tuple[str, float]:
    config = "--oem 1 --psm 6"
    try:
        data = pytesseract.image_to_data(img, lang="eng", config=config, output_type=Output.DICT)
        confs = [int(c) for c in data.get("conf", []) if c not in ("-1", -1)]
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    except Exception:
        avg_conf = 0.0
    text = pytesseract.image_to_string(img, lang="eng", config=config)
    return text, avg_conf


def page_text_with_ocr(page: "fitz.Page", dpi: int = 200) -> tuple[str, bool, float]:
    """Extract text with PyMuPDF first and fall back to OCR when needed.

    Returns: (text, used_ocr, ocr_confidence)
    """

    base_text = page.get_text("text", flags=fitz.TEXT_INHIBIT_SPACES)
    base_text_clean = base_text.strip()
    ascii_ok = _ascii_ratio(base_text_clean) >= 0.7

    need_ocr = len(base_text_clean) < 100 or not ascii_ok

    if not need_ocr:
        return base_text, False, 0.0

    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    ocr_text, ocr_conf = _ocr_image(image, dpi)

    # If OCR still looks poor, try a higher DPI once
    if (len(ocr_text.strip()) < len(base_text_clean)) or (ocr_conf < 45.0 and dpi < 300):
        try:
            pixmap2 = page.get_pixmap(dpi=300, alpha=False)
            image2 = Image.frombytes("RGB", [pixmap2.width, pixmap2.height], pixmap2.samples)
            ocr_text2, ocr_conf2 = _ocr_image(image2, 300)
            if len(ocr_text2.strip()) > len(ocr_text.strip()) or ocr_conf2 > ocr_conf:
                ocr_text, ocr_conf = ocr_text2, ocr_conf2
        except Exception:
            pass

    # Choose better between base and OCR
    chosen = ocr_text if len(ocr_text.strip()) >= len(base_text_clean) or not ascii_ok else base_text
    used_ocr = chosen is ocr_text
    return chosen, used_ocr, float(ocr_conf if used_ocr else 0.0)


def parse_pdf(
    pdf_path: Path,
    manifest_lookup: Dict[str, Any],
    base_meta: Dict[str, Any],
    out_handle,
) -> None:
    try:
        with pdf_path.open("rb") as file_handle:
            header = file_handle.read(8_192)
    except OSError as exc:
        print(f"[warn] cannot read {pdf_path}: {exc}", file=sys.stderr)
        return

    # Be permissive: attempt open even if header is odd
    if not header.startswith(b"%PDF-"):
        print(f"[info] odd PDF header, attempting open anyway: {pdf_path}", file=sys.stderr)

    # Try opening normally, then via stream if needed (sometimes repairs work)
    try:
        document = fitz.open(pdf_path)
    except Exception as e1:  # noqa: BLE001 - library can raise fitz specific errors
        try:
            data = pdf_path.read_bytes()
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as e2:  # noqa: BLE001
            print(
                f"[warn] cannot open {pdf_path}: normal open failed: {e1}; stream open failed: {e2}",
                file=sys.stderr,
            )
            return

    file_bytes = pdf_path.read_bytes()
    default_hash = sha256_bytes(file_bytes)

    manifest_record = manifest_lookup.get(pdf_path.name, {})
    merged_meta = dict(base_meta)
    merged_meta.update(
        {
            "filename": pdf_path.name,
            "file_path": str(pdf_path),
            "sha256": manifest_record.get("sha256", default_hash),
            "listed_date": manifest_record.get("listed_date"),
            "listed_date_raw": manifest_record.get("listed_date_raw"),
            "category": manifest_record.get("category"),
            "department": manifest_record.get("department"),
            "year_guess": manifest_record.get("year_guess"),
            "content_length": manifest_record.get("content_length", pdf_path.stat().st_size),
            "parsed_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    skip_toc = os.getenv("AUTOCOMPLY_SKIP_TOC", "false").lower() in {"1", "true", "yes"}
    for index in range(len(document)):
        page = document.load_page(index)
        try:
            text, used_ocr, ocr_conf = page_text_with_ocr(page)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] page parse failed p.{index+1} in {pdf_path}: {exc}", file=sys.stderr)
            continue
        # normalize & sanitize control chars
        text = text.replace("\x0c", "\n")
        text = re.sub(r"[ \t\r]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # detect simple Table of Contents pages and optionally skip
        is_toc = False
        if index <= 2:  # only first few pages typically
            header_lines = "\n".join(text.splitlines()[:10]).lower()
            if re.search(r"\btable of contents\b", header_lines) or re.match(r"^contents\b", header_lines):
                is_toc = True
        if is_toc and skip_toc:
            continue

        record = dict(merged_meta)
        record.update(
            {
                "page_no": index + 1,
                "page_count": len(document),
                "text": text,
                "is_ocr": used_ocr,
                "ocr_confidence": round(ocr_conf, 1) if used_ocr else None,
                "is_toc": is_toc,
            }
        )
        out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def handle_regulator_dir(industry_dir: Path, regulator_dir: Path) -> int:
    pdf_dir = regulator_dir / "pdf"
    if not pdf_dir.exists():
        return 0

    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_paths:
        return 0

    manifest_lookup = load_manifest(regulator_dir)
    output_path = OUT / f"{industry_dir.name}__{regulator_dir.name}.jsonl"
    with output_path.open("w", encoding="utf-8") as out_handle:
        base_meta = {
            "industry": industry_dir.name,
            "regulator": regulator_dir.name,
        }
        for pdf_path in pdf_paths:
            parse_pdf(pdf_path, manifest_lookup, base_meta, out_handle)

    print(f"[ok] wrote {output_path}")
    return len(pdf_paths)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pdf_count = 0

    for industry_dir in sorted(ROOT.glob("*")):
        if not industry_dir.is_dir():
            continue

        for regulator_dir in sorted(industry_dir.glob("*")):
            if not regulator_dir.is_dir():
                continue

            processed = handle_regulator_dir(industry_dir, regulator_dir)
            if processed == 0:
                print(f"[info] no PDFs under {regulator_dir}", file=sys.stderr)
                continue
            pdf_count += processed

    print(f"[done] processed {pdf_count} PDFs.")


if __name__ == "__main__":
    main()
