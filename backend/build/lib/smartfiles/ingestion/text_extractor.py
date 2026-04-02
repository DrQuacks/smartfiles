from __future__ import annotations

import pathlib
import string
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
        page_texts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            page_texts.append(text)
        joined = "\n".join(page_texts)

        # For downstream chunking and search, we preserve page
        # structure in the corpus by inserting lightweight markers
        # between pages. These markers are later used to derive page
        # ranges for each chunk.
        def _with_page_markers(texts: list[str]) -> str:
            parts: list[str] = []
            for idx, t in enumerate(texts, start=1):
                parts.append(f"[[[SMARTFILES_PAGE {idx}]]]")
                if t:
                    parts.append(t)
            return "\n".join(parts)

        joined_with_markers = _with_page_markers(page_texts)

        if joined.strip():
            # If the text layer exists but looks obviously garbled,
            # we can optionally try an OCR-based path as a fallback.
            if self._pdf_ocr_fallback and not _ocr_stack_available():
                return joined_with_markers
            if self._pdf_ocr_fallback and _is_probably_garbled(joined):
                ocr_text = self._pdf_ocr_fallback_for_pdf(path)
                # Only replace the text-layer output if OCR produced
                # something non-empty that looks at least as sane.
                if ocr_text.strip() and not _is_probably_garbled(ocr_text):
                    return ocr_text
            return joined_with_markers

        # If requested, fall back to OCR for PDFs with no text layer.
        if not self._pdf_ocr_fallback or not _ocr_stack_available():
            return joined
        # If no text layer is present, fall back to OCR.
        return self._pdf_ocr_fallback_for_pdf(path) or joined

    def _pdf_ocr_fallback_for_pdf(self, path: pathlib.Path) -> str:
        """Run the tiered OCR fallback for a PDF.

        This first performs a standard OCR pass and, if that fails to
        yield usable text, escalates to a stronger math-aware OCR pass
        with higher DPI and simple preprocessing.
        """

        # Tier 1: standard OCR on rendered pages.
        basic = _ocr_pages_for_pdf(path, dpi=200, strong=False)
        if basic.strip():
            return basic

        # Tier 2: strong OCR with preprocessing and math-aware config.
        strong_text = _ocr_pages_for_pdf(path, dpi=300, strong=True)
        if strong_text.strip():
            return strong_text

        return ""

    def _extract_image(self, path: pathlib.Path) -> str:
        if Image is None or pytesseract is None:
            # Graceful degradation if OCR stack is not installed
            return ""

        try:
            with Image.open(path) as img:  # type: ignore[call-arg]
                # Normalize to a mode that Tesseract handles well.
                img = img.convert("RGB")
                text = pytesseract.image_to_string(img)  # type: ignore[arg-type]

                # If the initial OCR result looks obviously garbled,
                # try a stronger math-aware OCR path with simple
                # preprocessing before giving up.
                if text and _is_probably_garbled(text):
                    strong_text = _ocr_image_strong_math(img)
                    if strong_text.strip() and not _is_probably_garbled(strong_text):
                        text = strong_text
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


def _ocr_stack_available() -> bool:
    return Image is not None and pytesseract is not None and convert_from_path is not None


def _ocr_pages_for_pdf(path: pathlib.Path, dpi: int, strong: bool) -> str:
    if convert_from_path is None or Image is None or pytesseract is None:
        return ""
    try:
        images = convert_from_path(str(path), dpi=dpi)  # type: ignore[arg-type]
    except Exception:
        return ""

    ocr_texts: list[str] = []
    for img in images:
        try:
            img = img.convert("RGB")
            if strong:
                # Lightweight preprocessing for scanned pages, plus a
                # math-aware OCR configuration when available.
                gray = img.convert("L")
                gray = ImageOps.autocontrast(gray)
                w, h = gray.size
                if max(w, h) < 2000:
                    scale = 1.5
                    gray = gray.resize((int(w * scale), int(h * scale)))
                ocr_img = gray
                config = "--oem 1 --psm 6 --dpi 300"
                try:
                    text = pytesseract.image_to_string(  # type: ignore[arg-type]
                        ocr_img,
                        lang="eng+equ",
                        config=config,
                    )
                except Exception:
                    text = pytesseract.image_to_string(ocr_img, config=config)  # type: ignore[arg-type]
            else:
                text = pytesseract.image_to_string(img)  # type: ignore[arg-type]
        except Exception:
            text = ""
        if text:
            ocr_texts.append(text)

    return "\n".join(ocr_texts) if ocr_texts else ""


def _ocr_image_strong_math(img: "Image.Image") -> str:  # type: ignore[name-defined]
    """Run a stronger, math-aware OCR pass on a PIL image.

    This is used as a fallback when the initial OCR text appears
    obviously garbled.
    """

    if Image is None or pytesseract is None:
        return ""

    try:
        gray = img.convert("L")
        gray = ImageOps.autocontrast(gray)
        w, h = gray.size
        if max(w, h) < 2000:
            scale = 1.5
            gray = gray.resize((int(w * scale), int(h * scale)))
        ocr_img = gray
        config = "--oem 1 --psm 6"
        try:
            text = pytesseract.image_to_string(  # type: ignore[arg-type]
                ocr_img,
                lang="eng+equ",
                config=config,
            )
        except Exception:
            text = pytesseract.image_to_string(ocr_img, config=config)  # type: ignore[arg-type]
    except Exception:
        text = ""

    return text or ""


def _is_probably_garbled(text: str) -> bool:
    """Heuristic to detect obviously garbled OCR/text-layer output.

    We keep this intentionally simple and conservative: only clearly
    junky text should trigger the heavier OCR fallback.
    """

    # Ignore very short snippets; they are hard to classify.
    non_ws = [ch for ch in text if not ch.isspace()]
    if len(non_ws) < 40:
        return False

    allowed = set(string.ascii_letters + string.digits + " .,;:!?-+*/=%()[]{}<>_^'\"\\|")
    good = sum(1 for ch in non_ws if ch in allowed)
    ratio = good / len(non_ws)

    if ratio < 0.6:
        return True

    # Treat a high density of obvious replacement characters as garbled.
    bad_markers = text.count("\ufffd")
    if bad_markers and bad_markers / len(non_ws) > 0.1:
        return True

    return False


def get_default_extractor() -> DefaultTextExtractor:
    # By default, enable the PDF OCR fallback (including the strong
    # secondary pass) so users get the best extraction behavior without
    # needing to pass extra flags.
    return DefaultTextExtractor(pdf_ocr_fallback=True)
