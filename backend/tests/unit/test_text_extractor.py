import pathlib

import pytest

from smartfiles.ingestion.text_extractor import DefaultTextExtractor


@pytest.mark.parametrize("paragraphs", [["First line", "Second line"], ["Only one"]])
def test_extract_docx_paragraphs(tmp_path, paragraphs):
    """DOCX extraction should join non-empty paragraphs with newlines."""

    try:
        import docx  # type: ignore
    except Exception:  # pragma: no cover - optional
        pytest.skip("python-docx is not available")

    path = tmp_path / "doc.docx"
    document = docx.Document()  # type: ignore[attr-defined]
    for text in paragraphs:
        document.add_paragraph(text)
    document.save(str(path))

    extractor = DefaultTextExtractor()
    extracted = extractor.extract_text(path)

    assert extracted == "\n".join(paragraphs)


def test_extract_pdf_uses_ocr_fallback_when_no_text(monkeypatch, tmp_path):
    """When the text layer is empty, the OCR fallback path is used."""

    from smartfiles.ingestion import text_extractor as te

    # Fake PdfReader that yields pages with no text.
    class FakePage:
        def extract_text(self):  # pragma: no cover - simple stub
            return ""

    class FakeReader:
        def __init__(self, *_args, **_kwargs):  # pragma: no cover
            self.pages = [FakePage(), FakePage()]

    monkeypatch.setattr(te.pypdf, "PdfReader", FakeReader)

    # Create a simple in-memory image for pdf2image to return.
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - optional
        pytest.skip("Pillow not available")

    img = Image.new("RGB", (100, 100), color="white")

    def fake_convert_from_path(_path, dpi=200):  # pragma: no cover - small helper
        # Ignore dpi here; tests focus on control flow.
        return [img]

    monkeypatch.setattr(te, "convert_from_path", fake_convert_from_path)

    # Make basic OCR return nothing and strong OCR return some text based
    # on the config argument.
    def fake_image_to_string(_image, config=None):  # pragma: no cover - simple stub
        if config and "--dpi 300" in str(config):
            return "STRONG OCR TEXT"
        return ""

    monkeypatch.setattr(te, "pytesseract", type("PT", (), {"image_to_string": staticmethod(fake_image_to_string)}))

    extractor = DefaultTextExtractor()
    pdf_path = tmp_path / "file.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    text = extractor.extract_text(pdf_path)

    assert "STRONG OCR TEXT" in text
