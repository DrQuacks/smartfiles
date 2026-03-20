from __future__ import annotations

import pathlib
from typing import Protocol

import pypdf

try:  # Optional dependencies for image OCR and PDF OCR fallback
    from PIL import Image, ImageFile, UnidentifiedImageError  # type: ignore
    import pytesseract  # type: ignore
    try:
        from pdf2image import convert_from_path  # type: ignore
    except Exception:  # pragma: no cover - optional
        convert_from_path = None  # type: ignore

    # Be tolerant of slightly truncated image files.
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional
    Image = None  # type: ignore
    ImageFile = None  # type: ignore
    UnidentifiedImageError = Exception  # type: ignore
    pytesseract = None  # type: ignore
    convert_from_path = None  # type: ignore


class TextExtractor(Protocol):
    def extract_text(self, path: pathlib.Path) -> str:  # pragma: no cover - protocol
        ...


class DefaultTextExtractor:
    """Extract text from PDFs and images (PNG/JPG) with optional OCR.

    If `pdf_ocr_fallback` is True, then for PDFs that yield no text via
    pypdf, we render pages to images and run Tesseract as a fallback.
    """

    def __init__(self, pdf_ocr_fallback: bool = False) -> None:
        self._pdf_ocr_fallback = pdf_ocr_fallback

    def extract_text(self, path: pathlib.Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        if suffix in {".png", ".jpg", ".jpeg"}:
            return self._extract_image(path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_pdf(self, path: pathlib.Path) -> str:
        # First try the fast text-layer extractor.
        reader = pypdf.PdfReader(str(path))
        texts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            texts.append(text)
        joined = "\n".join(texts)

        if joined.strip():
            return joined

        # If requested, fall back to OCR for PDFs with no text layer.
        if not self._pdf_ocr_fallback or Image is None or pytesseract is None or convert_from_path is None:
            return joined

        ocr_texts: list[str] = []
        try:
            images = convert_from_path(str(path))  # type: ignore[arg-type]
        except Exception:
            return joined

        for img in images:
            try:
                img = img.convert("RGB")
                ocr_text = pytesseract.image_to_string(img)  # type: ignore[arg-type]
            except Exception:
                ocr_text = ""
            if ocr_text:
                ocr_texts.append(ocr_text)

        if not ocr_texts:
            return joined
        return "\n".join(ocr_texts)

    def _extract_image(self, path: pathlib.Path) -> str:
        if Image is None or pytesseract is None:
            # Graceful degradation if OCR stack is not installed
            return ""

        try:
            with Image.open(path) as img:  # type: ignore[call-arg]
                # Normalize to a mode that Tesseract handles well.
                img = img.convert("RGB")
                text = pytesseract.image_to_string(img)  # type: ignore[arg-type]
        except UnidentifiedImageError:
            # PIL could not make sense of the bytes; let the caller
            # treat this as a no-text-extracted case.
            return ""
        except Exception:
            # Any other imaging/OCR error: also treat as no text.
            return ""

        return text or ""


def get_default_extractor(pdf_ocr_fallback: bool = False) -> DefaultTextExtractor:
    return DefaultTextExtractor(pdf_ocr_fallback=pdf_ocr_fallback)
