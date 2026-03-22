import os
from pathlib import Path

from smartfiles.ingestion.indexer import extract_documents, build_index_from_corpus, chunk_corpus_from_text
from smartfiles.database.text_store import get_corpus_dir, get_stats_dir, get_chunks_dir, save_document_text
from smartfiles.folder_registry import ensure_folder_entry


def test_extract_documents_writes_corpus_and_stats(tmp_path, monkeypatch):
    """End-to-end extraction should write corpus files and a stats file.

    This test stubs out the heavy format-specific extraction and focuses on
    verifying that `extract_documents` drives the folder registry, text
    store, and stats logging correctly.
    """

    # Use a temp data dir so we don't touch the real ~/.smartfiles.
    data_dir = tmp_path / "data"
    monkeypatch.setenv("SMARTFILES_DATA_DIR", str(data_dir))

    # Create a synthetic root folder with a couple of supported files.
    root = tmp_path / "root"
    root.mkdir()
    pdf = root / "doc1.pdf"
    jpg = root / "image1.jpg"
    pdf.write_bytes(b"%PDF-1.4")
    jpg.write_bytes(b"binary")

    # Stub out the extractor so we don't rely on real PDF/image parsing.
    from smartfiles import ingestion as ingestion_pkg

    class FakeExtractor:
        def __init__(self):
            self.calls = []

        def extract_text(self, path: Path) -> str:  # pragma: no cover - trivial stub
            self.calls.append(path)
            return f"TEXT FROM {path.name}"

    # Reuse a single extractor instance so we can verify that running
    # extraction again without recreate_text=True does not rerun
    # expensive per-file extraction work when corpus files already exist.
    extractor_instance = FakeExtractor()

    def fake_get_default_extractor():  # pragma: no cover - trivial stub
        return extractor_instance

    monkeypatch.setattr(ingestion_pkg.indexer, "get_default_extractor", fake_get_default_extractor)

    # Run extraction the first time, recreating the corpus.
    extract_documents(root_folder=root, recreate_text=True)

    # Resolve the per-root corpus/stats directories via the helpers.
    entry = ensure_folder_entry(root)
    run_root = Path(entry.path)
    corpus_dir = get_corpus_dir(run_root)
    stats_dir = get_stats_dir(run_root)

    # Corpus should contain one .txt file per input document.
    corpus_files = sorted(p.name for p in corpus_dir.rglob("*.txt"))
    assert corpus_files == ["doc1.pdf.txt", "image1.jpg.txt"]

    # Each corpus file should contain the stubbed text.
    for name in corpus_files:
        text = (corpus_dir / name).read_text(encoding="utf-8")
        assert "TEXT FROM" in text

    # Stats directory should contain exactly one extraction_XXXX.txt file
    # with a Summary section.
    stats_files = list(stats_dir.glob("extraction_*.txt"))
    assert len(stats_files) == 1
    stats_text = stats_files[0].read_text(encoding="utf-8")
    assert "Summary:" in stats_text
    assert "[OK]" in stats_text

    # Run extraction a second time without recreating text; this should
    # reuse existing corpus files instead of rerunning extraction.
    extract_documents(root_folder=root, recreate_text=False)

    # We have two input files, so the extractor should have been called
    # exactly twice across both runs (once per file), not four times.
    assert len(extractor_instance.calls) == 2


def test_build_index_from_corpus_uses_saved_text(tmp_path, monkeypatch):
    """`build_index_from_corpus` should read from the corpus and call
    the vector store with embedded chunks.

    We stub out the embedding model and vector store to avoid heavy
    dependencies while still exercising chunking and corpus loading.
    """

    # Use a temp data dir for this test run.
    data_dir = tmp_path / "data"
    monkeypatch.setenv("SMARTFILES_DATA_DIR", str(data_dir))

    # Prepare a root folder and populate the corpus via text_store.
    root = tmp_path / "root"
    root.mkdir()
    entry = ensure_folder_entry(root)
    run_root = Path(entry.path)

    original_path = root / "notes.docx"
    original_path.write_text("dummy", encoding="utf-8")
    # Save some text for this file in the corpus.
    sample_text = "Gravity is a force between masses."
    save_document_text(run_root, original_path, sample_text)

    # Stub the embedding model and vector store used by the indexer.
    from smartfiles.ingestion import indexer as indexer_mod

    class FakeEmbedder:
        def __init__(self):
            self.seen_texts = []

        def embed_texts(self, texts):  # pragma: no cover - simple stub
            # Record the texts we're asked to embed and return dummy vectors.
            self.seen_texts.extend(texts)
            return [[0.0] * 3 for _ in texts]

    class FakeVectorStore:
        def __init__(self):
            self.add_calls = []

        def add_documents(self, *, chunks, embeddings):  # pragma: no cover - simple stub
            self.add_calls.append({"chunks": chunks, "embeddings": embeddings})

    fake_embedder = FakeEmbedder()
    fake_store = FakeVectorStore()

    def fake_get_default_embedding_model():  # pragma: no cover
        return fake_embedder

    def fake_get_default_vector_store(recreate=False):  # pragma: no cover
        return fake_store

    monkeypatch.setattr(indexer_mod, "get_default_embedding_model", fake_get_default_embedding_model)
    monkeypatch.setattr(indexer_mod, "get_default_vector_store", fake_get_default_vector_store)

    # Run index build from the prepared corpus (saving chunks by default).
    build_index_from_corpus(root_folder=run_root, recreate_index=True)

    # We should have exactly one add_documents call with at least one chunk
    # whose text contains our sample text.
    assert len(fake_store.add_calls) == 1
    call = fake_store.add_calls[0]
    chunks = call["chunks"]
    assert chunks
    combined_text = "\n".join(c.text for c in chunks)
    assert "Gravity is a force" in combined_text

    # Chunks directory should contain at least one chunk file for this document.
    chunks_dir = get_chunks_dir(run_root)
    chunk_files = list(chunks_dir.rglob("*.txt"))
    assert chunk_files


def test_chunk_corpus_from_text_writes_chunks_without_embedding(tmp_path, monkeypatch):
    """`chunk_corpus_from_text` should read from the corpus and write
    chunk files without requiring embeddings or the vector store.

    This lets us inspect chunking independently of the embedding/index
    stages.
    """

    # Use a temp data dir for this test run.
    data_dir = tmp_path / "data"
    monkeypatch.setenv("SMARTFILES_DATA_DIR", str(data_dir))

    # Prepare a root folder and populate the corpus via text_store.
    root = tmp_path / "root"
    root.mkdir()
    entry = ensure_folder_entry(root)
    run_root = Path(entry.path)

    original_path = root / "doc.txt"
    original_path.write_text("one two three four five six", encoding="utf-8")

    # Save some text for this file in the corpus.
    sample_text = "Gravity is a force between masses. It acts at a distance."
    save_document_text(run_root, original_path, sample_text)

    # Run chunking from the prepared corpus.
    chunk_corpus_from_text(root_folder=run_root, save_chunks=True)

    # Chunks directory should contain at least one chunk file for this document.
    chunks_dir = get_chunks_dir(run_root)
    chunk_files = list(chunks_dir.rglob("*.txt"))
    assert chunk_files

    # Each chunk file should contain a portion of the sample text.
    combined = "\n".join(p.read_text(encoding="utf-8") for p in chunk_files)
    assert "Gravity is a force" in combined
