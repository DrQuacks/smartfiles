from __future__ import annotations

import pathlib
from typing import Protocol

import pypdf

try:  # Optional dependencies for image OCR and PDF OCR fallback
    from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError  # type: ignore
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

try:  # Optional dependency for DOCX parsing
    import docx  # type: ignore
except Exception:  # pragma: no cover - optional
    docx = None  # type: ignore


class TextExtractor(Protocol):
    def extract_text(self, path: pathlib.Path) -> str:  # pragma: no cover - protocol
        ...


class DefaultTextExtractor:
    """Extract text from PDFs and images (PNG/JPG) with OCR.

    For PDFs we always try pypdf's text layer first. If that yields no
    text, we first try a standard OCR pass and, if that still produces
    no text, automatically fall back to a stronger OCR path with higher
    DPI rendering and simple preprocessing.
    """

    def __init__(self, pdf_ocr_fallback: bool = True) -> None:
        # We keep `pdf_ocr_fallback` for flexibility, but the default
        # is to enable the OCR fallback for PDFs.
        self._pdf_ocr_fallback = pdf_ocr_fallback

    def extract_text(self, path: pathlib.Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        if suffix in {".png", ".jpg", ".jpeg"}:
            return self._extract_image(path)
        if suffix == ".docx":
            return self._extract_docx(path)
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

        # Tier 1: standard OCR on rendered pages.
        def _ocr_pages(dpi: int, strong: bool) -> str:
            try:
                images = convert_from_path(str(path), dpi=dpi)  # type: ignore[arg-type]
            except Exception:
                return ""

            ocr_texts: list[str] = []
            for img in images:
                try:
                    img = img.convert("RGB")
                    if strong:
                        # Lightweight preprocessing for scanned pages.
                        gray = img.convert("L")
                        gray = ImageOps.autocontrast(gray)
                        w, h = gray.size
                        if max(w, h) < 2000:
                            scale = 1.5
                            gray = gray.resize((int(w * scale), int(h * scale)))
                        ocr_img = gray
                        config = "--oem 1 --psm 6 --dpi 300"
                        text = pytesseract.image_to_string(ocr_img, config=config)  # type: ignore[arg-type]
                    else:
                        text = pytesseract.image_to_string(img)  # type: ignore[arg-type]
                except Exception:
                    text = ""
                if text:
                    ocr_texts.append(text)

            return "\n".join(ocr_texts) if ocr_texts else ""

        # First, try a standard OCR pass.
        basic = _ocr_pages(dpi=200, strong=False)
        if basic.strip():
            return basic

        # If that fails to produce any text, escalate to strong OCR.
        strong_text = _ocr_pages(dpi=300, strong=True)
        if strong_text.strip():
            return strong_text

        return joined

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

    def _extract_docx(self, path: pathlib.Path) -> str:
        """Extract text from a .docx file.

        If python-docx is not installed, we degrade gracefully and
        return an empty string so the caller can treat this as a
        no-text-extracted case.
        """

        if docx is None:
            return ""

        try:
            document = docx.Document(str(path))  # type: ignore[arg-type]
        except Exception:
            return ""

        parts: list[str] = []
        for para in document.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        return "\n".join(parts)


def get_default_extractor() -> DefaultTextExtractor:
    # By default, enable the PDF OCR fallback (including the strong
    # secondary pass) so users get the best extraction behavior without
    # needing to pass extra flags.
    return DefaultTextExtractor(pdf_ocr_fallback=True)
