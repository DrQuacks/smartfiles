# SmartFiles Embedding Models

This backend uses SentenceTransformers-compatible embedding models. The
actual model can be swapped via environment variables so you can
experiment with different local, open source models during development.

## Configuration

Selection precedence (highest to lowest):

1. `SMARTFILES_EMBEDDING_MODEL` – explicit model id or local filesystem path.
2. `SMARTFILES_EMBEDDING_PROFILE` – logical profile key from the table below.
3. Built-in default profile `bge-small-en-v1`.

Examples:

```bash
# Use a built-in profile
export SMARTFILES_EMBEDDING_PROFILE=bge-base-en-v1

# Or point directly at a local model directory
export SMARTFILES_EMBEDDING_MODEL=/path/to/local/bge-small-en-v1
```

## Supported Profiles

These profiles are defined in `smartfiles/embeddings/embedding_model.py`
and are intended as sensible defaults for local / OSS use:

| Profile key          | Hugging Face id                           | Notes                                           |
|----------------------|-------------------------------------------|------------------------------------------------|
| `bge-small-en-v1`    | `BAAI/bge-small-en-v1`                    | Default; 768d English model, good speed/quality|
| `bge-base-en-v1`     | `BAAI/bge-base-en-v1`                     | Larger 1024d English model for higher quality  |
| `all-minilm-l6-v2`   | `sentence-transformers/all-MiniLM-L6-v2`  | Very small 384d model, fast and lightweight    |

All of these can be used fully offline by downloading the model once
(e.g. with `git lfs` or `huggingface-cli`), placing it on disk, and
pointing `SMARTFILES_EMBEDDING_MODEL` at the local directory.
