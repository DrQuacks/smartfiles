# SmartFiles

SmartFiles is a **local AI-powered search engine for your document folders**.

Instead of manually digging through directories, SmartFiles lets you search your documents semantically:

> "precalc worksheet from Saint Ignatius"

and instantly retrieves the most relevant files.

Everything runs **locally on your machine**. No cloud services or external APIs are required.

---

## Features

- Semantic search over documents
- Works with PDFs, Word docs, and images
- Local vector database
- Finder-style search workflow
- Incremental indexing
- No cloud dependencies

Planned features:

- OCR for scanned documents
- Automatic document tagging
- Finder-style preview UI
- Background indexing

---

## Example

Search query:

precalc worksheet SI

SmartFiles might return:

92  SI_precalc_review.pdf
87  Saint_Ignatius_trig_homework.pdf
65  limits_practice.pdf

Results are ranked by semantic similarity and metadata signals.

---

## Installation

Development installation:

pip install smartfiles

Future installation options:

brew install smartfiles

Desktop app:

SmartFiles.app

---

## Quick Start

Index a folder:

smartfiles index ~/Documents/Tutoring

Search your documents:

smartfiles search "precalculus worksheet"

Launch the web interface:

smartfiles search

---

## Architecture

Documents
 ↓
Text extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector database
 ↓
Search engine

See the docs/ folder for detailed architecture documentation.

---

## Project Structure

smartfiles/

backend/
frontend/
docs/

---

## Development

Backend server:

uvicorn smartfiles.server.api:app --reload

Frontend:

npm run dev

---

## Roadmap

See docs/roadmap.md

---

## License

MIT
