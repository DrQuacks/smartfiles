import pathlib

import typer

from smartfiles.ingestion.indexer import (
    run_indexing_pipeline,
    extract_documents,
    build_index_from_corpus,
)
from smartfiles.search.search_engine import run_search

app = typer.Typer(help="SmartFiles CLI: index and search your documents.")


@app.command()
def extract(
    folder: pathlib.Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, help="Folder to parse"),
    recreate_text: bool = typer.Option(
        False,
        "--recreate-text",
        help="Delete and rebuild the raw text corpus before extracting.",
    ),
    pdf_ocr: bool = typer.Option(
        False,
        "--pdf-ocr",
        help="For PDFs with no text layer, fall back to OCR (slower).",
    ),
):
    """Only parse documents and write raw text files to the corpus."""
    extract_documents(root_folder=folder, recreate_text=recreate_text, pdf_ocr=pdf_ocr)


@app.command()
def index_from_text(
    folder: pathlib.Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, help="Folder whose extracted text should be indexed"),
    recreate: bool = typer.Option(False, "--recreate", help="Recreate the vector index before indexing."),
):
    """Chunk, embed, and index using the existing raw text corpus."""
    build_index_from_corpus(root_folder=folder, recreate_index=recreate)


@app.command()
def index(
    folder: pathlib.Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, help="Folder to index"),
    recreate: bool = typer.Option(False, "--recreate", help="Recreate both corpus and index from scratch"),
    pdf_ocr: bool = typer.Option(
        False,
        "--pdf-ocr",
        help="For PDFs with no text layer, fall back to OCR during extraction.",
    ),
):
    """Run the full pipeline: extract text, chunk, embed, and index."""
    run_indexing_pipeline(root_folder=folder, recreate=recreate, pdf_ocr=pdf_ocr)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(5, "-k", "--top-k", help="Number of results to return"),
):
    """Search the indexed documents semantically."""
    results = run_search(query=query, k=k)
    if not results:
        typer.echo("No results found.")
        raise typer.Exit(code=0)

    for rank, result in enumerate(results, start=1):
        score = result.get("score", 0)
        path = result.get("filepath", "?")
        snippet = result.get("text", "").replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        typer.echo(f"{rank:2d}. [{score:5.1f}] {path}\n    {snippet}")


if __name__ == "__main__":  # pragma: no cover
    app()
