import pathlib

import typer

from smartfiles.ingestion.indexer import (
    run_indexing_pipeline,
    extract_documents,
    build_index_from_corpus,
    chunk_corpus_from_text,
)
from smartfiles.search.search_engine import run_search
from smartfiles.embeddings.embedding_model import get_default_embedding_model
from smartfiles.database.vector_store import get_default_vector_store

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
        page_info = ""
        page_start = result.get("page_start")
        page_end = result.get("page_end")
        if isinstance(page_start, int):
            if isinstance(page_end, int) and page_end != page_start:
                page_info = f" (pages {page_start}-{page_end})"
            else:
                page_info = f" (page {page_start})"
        snippet = result.get("text", "").replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        typer.echo(f"{rank:2d}. [{score:5.1f}] {path}{page_info}\n    {snippet}")


@app.command()
def debug_scores(
    query: str = typer.Argument(..., help="Search query to debug"),
    k: int = typer.Option(10, "-k", "--top-k", help="Number of results to inspect"),
):
    """Run a query and log raw distances and cosine-like scores.

    This command uses the same embedder and vector store as the
    regular search path, but enables additional logging so you can
    manually inspect:

    - the raw distances returned by Chroma
    - the derived similarity s = 1 - d (clamped to [-1, 1])
    - the final score = (s + 1) / 2 * 100

    Enable more verbose internal logging by setting the
    SMARTFILES_DEBUG_SCORES=1 environment variable when running this
    command.
    """

    embedder = get_default_embedding_model()
    store = get_default_vector_store(recreate=False)

    embedding = embedder.embed_texts([query])[0]
    results = store.search(query_embedding=embedding, k=k)

    if not results:
        typer.echo("No results found.")
        raise typer.Exit(code=0)

    for rank, result in enumerate(results, start=1):
        score = result.get("score", 0.0)
        path = result.get("filepath", "?")
        dist = result.get("distance")
        sim = None
        if dist is not None:
            try:
                raw_dist = float(dist)
                sim = 1.0 - raw_dist
            except Exception:
                sim = None

        typer.echo(f"rank={rank:2d} score={score:6.2f} filepath={path}")
        if dist is not None:
            typer.echo(f"    raw_dist={dist!r} sim_from_dist={sim}")


    @app.command("benchmark-beir")
    def benchmark_beir(
        dataset: str = typer.Argument(..., help="BEIR dataset name, e.g. 'scifact'"),
        split: str = typer.Option(
            "test",
            "--split",
            help="Dataset split to use (typically 'test', 'dev', or 'train')",
        ),
        top_k: int = typer.Option(
            10,
            "-k",
            "--top-k",
            help="Maximum cutoff K for evaluation metrics",
        ),
        batch_size: int = typer.Option(
            128,
            "--batch-size",
            help="Batch size when embedding the BEIR corpus",
        ),
        skip_index: bool = typer.Option(
            False,
            "--skip-index",
            help="Reuse an existing BEIR index instead of rebuilding it",
        ),
        run_tag: str = typer.Option(
            "",
            "--tag",
            help="Optional label to attach to this benchmark run",
        ),
    ):
        """Run a BEIR benchmark using the current SmartFiles stack.

        This command is optional and requires installing the 'benchmark'
        extra: `pip install .[benchmark]` from the backend directory.

        All benchmark data and indexes are stored under
        `SMARTFILES_DATA_DIR/benchmarks/beir` and do not affect your
        normal SmartFiles corpus or index.
        """

        try:
            from smartfiles.benchmarks.beir_runner import run_beir_benchmark
        except Exception as exc:  # pragma: no cover - defensive guard
            typer.echo(
                "Error: BEIR benchmarking dependencies are not available.\n"
                "Install the 'benchmark' extra from the backend directory, e.g.:\n"
                "  pip install .[benchmark]",
                err=True,
            )
            raise typer.Exit(code=1) from exc

        run_beir_benchmark(
            dataset_name=dataset,
            split=split,
            top_k=top_k,
            batch_size=batch_size,
            skip_index=skip_index,
            run_tag=run_tag or None,
        )


    @app.command("build-dimdrop-mask-beir")
    def build_dimdrop_mask_beir(
        dataset: str = typer.Argument(..., help="BEIR dataset name, e.g. 'scifact'"),
        split: str = typer.Option(
            "test",
            "--split",
            help="Dataset split used when indexing the BEIR corpus",
        ),
        sample_size: int = typer.Option(
            2000,
            "--sample-size",
            help="Number of BEIR embeddings sampled to estimate dim variances",
        ),
        batch_size: int = typer.Option(
            128,
            "--batch-size",
            help="Batch size used when (re)indexing the BEIR corpus",
        ),
        reindex: bool = typer.Option(
            False,
            "--reindex",
            help="Rebuild BEIR index before computing dim-order mask",
        ),
        output: str = typer.Option(
            "",
            "--output",
            help="Optional output path for dimdrop_dim_order.npy",
        ),
    ):
        """Build a BEIR-based dim-drop mask artifact (.npy).

        The generated mask can be consumed by the API via either:
        - SMARTFILES_DIMDROP_MASK_PATH=<absolute path>, or
        - SMARTFILES_DIMDROP_BEIR_DATASET=<dataset name>
        """

        try:
            from pathlib import Path
            from smartfiles.benchmarks.dimdrop_mask import build_beir_dimdrop_mask
        except Exception as exc:  # pragma: no cover
            typer.echo(
                "Error: Unable to import dim-drop mask builder.",
                err=True,
            )
            raise typer.Exit(code=1) from exc

        output_path = Path(output.strip()).expanduser().resolve() if output.strip() else None

        try:
            npy_path, _meta_path = build_beir_dimdrop_mask(
                dataset=dataset,
                split=split,
                sample_size=sample_size,
                batch_size=batch_size,
                reindex=reindex,
                output_path=output_path,
            )
            typer.echo(f"Built dim-drop mask: {npy_path}")
            typer.echo(f"Set SMARTFILES_DIMDROP_MASK_PATH={npy_path} to use it in API")
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc


    @app.command("build-dimdrop-mask-beir-sampled")
    def build_dimdrop_mask_beir_sampled(
        datasets: str = typer.Argument(
            ...,
            help="Comma-separated BEIR datasets, e.g. 'scifact,nfcorpus,fiqa,trec-covid'",
        ),
        split: str = typer.Option(
            "test",
            "--split",
            help="Dataset split used when sampling raw BEIR texts",
        ),
        per_dataset_sample_size: int = typer.Option(
            500,
            "--per-dataset-sample-size",
            help="Maximum raw documents sampled per dataset before embedding",
        ),
        batch_size: int = typer.Option(
            128,
            "--batch-size",
            help="Embedding batch size used while building the mask",
        ),
        seed: int = typer.Option(
            13,
            "--seed",
            help="Random seed used for dataset sampling",
        ),
        label: str = typer.Option(
            "",
            "--label",
            help="Optional label used for the output artifact directory",
        ),
        output: str = typer.Option(
            "",
            "--output",
            help="Optional explicit output path for dimdrop_dim_order.npy",
        ),
    ):
        """Build a BEIR-based dim-drop mask from sampled raw texts only.

        This avoids indexing whole datasets into Chroma. It is the
        recommended workflow for exploratory variance masks.
        """

        try:
            from pathlib import Path
            from smartfiles.benchmarks.dimdrop_mask import build_beir_sampled_dimdrop_mask
        except Exception as exc:  # pragma: no cover
            typer.echo(
                "Error: Unable to import sampled dim-drop mask builder.",
                err=True,
            )
            raise typer.Exit(code=1) from exc

        dataset_list = [d.strip() for d in datasets.split(",") if d.strip()]
        output_path = Path(output.strip()).expanduser().resolve() if output.strip() else None

        try:
            npy_path, _meta_path = build_beir_sampled_dimdrop_mask(
                datasets=dataset_list,
                split=split,
                per_dataset_sample_size=per_dataset_sample_size,
                batch_size=batch_size,
                seed=seed,
                output_path=output_path,
                label=label.strip() or None,
            )
            typer.echo(f"Built sampled dim-drop mask: {npy_path}")
            typer.echo(f"Set SMARTFILES_DIMDROP_MASK_PATH={npy_path} to use it in API")
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc


    @app.command("build-dimdrop-mask-mixed-sampled")
    def build_dimdrop_mask_mixed_sampled(
        beir: list[str] = typer.Option(
            [],
            "--beir",
            help="Repeatable BEIR dataset name to sample from, e.g. --beir fiqa --beir nfcorpus",
        ),
        hf: list[str] = typer.Option(
            [],
            "--hf",
            help=(
                "Repeatable Hugging Face dataset spec using repo_id::config::split::text_field, "
                "e.g. --hf fancyzhx/ag_news --hf google/wiki40b::en::train::text"
            ),
        ),
        local: list[str] = typer.Option(
            [],
            "--local",
            help="Repeatable local root folder whose extracted raw-text corpus should be sampled",
        ),
        include_registered_local: bool = typer.Option(
            False,
            "--include-registered-local",
            help="Also sample from all folders in the SmartFiles registry",
        ),
        beir_split: str = typer.Option(
            "test",
            "--beir-split",
            help="Split used when sampling raw BEIR texts",
        ),
        per_source_sample_size: int = typer.Option(
            500,
            "--per-source-sample-size",
            help="Maximum raw documents sampled per source before embedding",
        ),
        hf_max_scan_examples: int = typer.Option(
            20000,
            "--hf-max-scan-examples",
            help="Maximum streaming rows scanned per HF source (caps download/work)",
        ),
        batch_size: int = typer.Option(
            128,
            "--batch-size",
            help="Embedding batch size used while building the mask",
        ),
        seed: int = typer.Option(
            13,
            "--seed",
            help="Random seed used for source sampling",
        ),
        label: str = typer.Option(
            "",
            "--label",
            help="Optional label used for the output artifact directory",
        ),
        output: str = typer.Option(
            "",
            "--output",
            help="Optional explicit output path for dimdrop_dim_order.npy",
        ),
    ):
        """Build a mixed dim-drop mask from local, BEIR, and HF raw texts.

        This avoids Chroma indexing completely and is the preferred way
        to bootstrap a broader mask corpus after the failed full-index
        approach.
        """

        try:
            from pathlib import Path
            from smartfiles.benchmarks.dimdrop_mask import build_mixed_sampled_dimdrop_mask
        except Exception as exc:  # pragma: no cover
            typer.echo(
                "Error: Unable to import mixed sampled dim-drop mask builder.",
                err=True,
            )
            raise typer.Exit(code=1) from exc

        output_path = Path(output.strip()).expanduser().resolve() if output.strip() else None

        try:
            npy_path, _meta_path = build_mixed_sampled_dimdrop_mask(
                beir_datasets=beir,
                hf_datasets=hf,
                local_folders=local,
                include_registered_local=include_registered_local,
                beir_split=beir_split,
                per_source_sample_size=per_source_sample_size,
                hf_max_scan_examples=hf_max_scan_examples,
                batch_size=batch_size,
                seed=seed,
                output_path=output_path,
                label=label.strip() or None,
            )
            typer.echo(f"Built mixed sampled dim-drop mask: {npy_path}")
            typer.echo(f"Set SMARTFILES_DIMDROP_MASK_PATH={npy_path} to use it in API")
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc


if __name__ == "__main__":  # pragma: no cover
    app()
