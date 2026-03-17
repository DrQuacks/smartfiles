from __future__ import annotations

import pathlib
from typing import Protocol

import pypdf

try:  # Optional dependencies for image OCR
    from PIL import Image  # type: ignore
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional
    Image = None  # type: ignore
    pytesseract = None  # type: ignore


class TextExtractor(Protocol):
    def extract_text(self, path: pathlib.Path) -> str:  # pragma: no cover - protocol
        ...


class DefaultTextExtractor:
    """Extract text from PDFs and images (PNG/JPG) with optional OCR."""

    def extract_text(self, path: pathlib.Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        if suffix in {".png", ".jpg", ".jpeg"}:
            return self._extract_image(path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_pdf(self, path: pathlib.Path) -> str:
        reader = pypdf.PdfReader(str(path))
        texts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            texts.append(text)
        return "\n".join(texts)

    def _extract_image(self, path: pathlib.Path) -> str:
        if Image is None or pytesseract is None:
            # Graceful degradation if OCR stack is not installed
            return ""
        with Image.open(path) as img:  # type: ignore[call-arg]
            text = pytesseract.image_to_string(img)  # type: ignore[arg-type]
        return text or ""


def get_default_extractor() -> DefaultTextExtractor:
    return DefaultTextExtractor()
