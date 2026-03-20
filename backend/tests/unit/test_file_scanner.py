import pathlib

from smartfiles.ingestion.file_scanner import list_files


def test_list_files_filters_supported_extensions(tmp_path):
    root = tmp_path
    # Supported
    (root / "a.pdf").write_bytes(b"%PDF-1.4")
    (root / "b.JPG").write_bytes(b"data")
    (root / "c.docx").write_bytes(b"PK")
    # Unsupported
    (root / "d.txt").write_text("hello", encoding="utf-8")
    (root / "e.doc").write_bytes(b"old word")

    files = list_files(root)
    names = sorted(p.name for p in files)

    assert names == ["a.pdf", "b.JPG", "c.docx"]
