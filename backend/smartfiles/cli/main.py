import pathlib

import typer

from smartfiles.ingestion.indexer import (
    run_indexing_pipeline,
    extract_documents,
    build_index_from_corpus,
    chunk_corpus_from_text,
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
):
    """Only parse documents and write raw text files to the corpus."""
    extract_documents(root_folder=folder, recreate_text=recreate_text)


@app.command()
def index_from_text(
    folder: pathlib.Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, help="Folder whose extracted text should be indexed"),
    recreate: bool = typer.Option(False, "--recreate", help="Recreate the vector index before indexing."),
    save_chunks: bool = typer.Option(
        True,
        "--save-chunks/--no-save-chunks",
        help="Write per-chunk text files under the 'chunks' folder for inspection.",
    ),
    chunk_size: int = typer.Option(
        500,
        "--chunk-size",
        min=1,
        help="Approximate number of words per chunk.",
    ),
    overlap: int = typer.Option(
        50,
        "--chunk-overlap",
        min=0,
        help="Approximate number of overlapping words between chunks.",
    ),
):
    """Chunk, embed, and index using the existing raw text corpus."""
    build_index_from_corpus(
        root_folder=folder,
        recreate_index=recreate,
        save_chunks=save_chunks,
        chunk_size=chunk_size,
        overlap=overlap,
    )


@app.command()
def chunk_from_text(
    folder: pathlib.Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Folder whose extracted text should be chunked",
    ),
    save_chunks: bool = typer.Option(
        True,
        "--save-chunks/--no-save-chunks",
        help="Write per-chunk text files under the 'chunks' folder for inspection.",
    ),
):
    """Chunk documents using the existing raw text corpus, without embedding."""
    chunk_corpus_from_text(
        root_folder=folder,
        save_chunks=save_chunks,
    )


@app.command()
def index(
    folder: pathlib.Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, help="Folder to index"),
    recreate: bool = typer.Option(False, "--recreate", help="Recreate both corpus and index from scratch"),
    save_chunks: bool = typer.Option(
        True,
        "--save-chunks/--no-save-chunks",
        help="Write per-chunk text files under the 'chunks' folder for inspection.",
    ),
    chunk_size: int = typer.Option(
        500,
        "--chunk-size",
        min=1,
        help="Approximate number of words per chunk.",
    ),
    overlap: int = typer.Option(
        50,
        "--chunk-overlap",
        min=0,
        help="Approximate number of overlapping words between chunks.",
    ),
):
    """Run the full pipeline: extract text, chunk, embed, and index."""
    run_indexing_pipeline(
        root_folder=folder,
        recreate=recreate,
        save_chunks=save_chunks,
        chunk_size=chunk_size,
        overlap=overlap,
    )


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
