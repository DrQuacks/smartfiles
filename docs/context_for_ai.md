# SmartFiles — AI Context Document

This file provides context for AI coding assistants (GitHub Copilot, ChatGPT, etc.) to understand the architecture and goals of the SmartFiles project.

SmartFiles is a **local AI-powered semantic search engine for document folders**.

The goal is to allow users to search their documents using natural language queries such as:

"precalc worksheet from Saint Ignatius"

and receive a ranked list of the most relevant documents.

Everything runs **locally**. No cloud services or external APIs are required.

---

# Core Concept

SmartFiles converts documents into **vector embeddings** and stores them in a **local vector database**.

When the user searches, the query is embedded and the database retrieves the most semantically similar document chunks.

The system is conceptually similar to a **local RAG retrieval pipeline**, but without a generative LLM.

---

# High-Level Architecture

Documents flow through the following pipeline:

Files
↓
Text Extraction
↓
Chunking
↓
Embedding
↓
Vector Database
↓
Search Engine
↓
UI

---

# Key Technologies

Backend:

Python

Important libraries:

Typer → CLI interface  
FastAPI → backend API  
Sentence Transformers → embedding model  
Chroma → vector database  
pypdf → PDF text extraction  
pytesseract → OCR fallback

Frontend:

React + TypeScript

PDF previews will use:

PDF.js

---

# Project Structure

Repository layout:
smartfiles/

backend/
smartfiles/
cli/
ingestion/
embeddings/
database/
search/
server/

frontend/

docs/


Backend modules are organized by responsibility.

---

# Backend Module Responsibilities

## cli/

Command line interface.

Example commands:

smartfiles extract ~/Documents  
smartfiles index-from-text ~/Documents  
smartfiles index ~/Documents  
smartfiles search "precalc worksheet"

Implemented using Typer.

---

## ingestion/

Document processing pipeline.

Responsible for:

• scanning folders  
• extracting text  
• chunking documents  
• extracting metadata  

Key modules:

file_scanner.py  
text_extractor.py  
chunker.py  
metadata_extractor.py  
indexer.py

---

## embeddings/

Handles embedding generation.

embedding_model.py

Responsibilities:

• load embedding model  
• convert text → embedding vectors  

Initial model:

bge-small-en

Using sentence-transformers.

---

## database/

Vector database abstraction.

vector_store.py

This wraps Chroma so the rest of the system does not depend directly on Chroma APIs.

Responsibilities:

• store embeddings  
• perform vector search  
• delete documents  

All vector operations go through this module.

---

## search/

Search logic.

search_engine.py  
reranker.py

Search pipeline:

query  
↓  
embed query  
↓  
vector search  
↓  
rerank results  
↓  
return ranked documents

Ranking combines:

• semantic similarity  
• metadata signals  
• recency

---

## server/

FastAPI backend for the UI.

api.py

Endpoints:

POST /search  
GET /document  
GET /preview

---

# Indexing Pipeline

The indexing pipeline converts documents into searchable embeddings.

Steps:

1. Scan folders for supported files

Supported formats (initial implementation):

PDF  
PNG  
JPG

Planned (not yet implemented):

DOCX

2. Extract text

Primary method:

pypdf

Fallback:

OCR using pytesseract

3. Clean text

Normalize whitespace and remove formatting artifacts.

4. Chunk documents

Typical parameters:

chunk_size = 500 tokens  
overlap = 50 tokens

Small worksheets may use page-level chunks.

5. Extract metadata

Examples:

school  
course  
topic  
filename  
folder

Example rule:

If "Saint Ignatius" appears → school = SI

6. Generate embeddings

embedding = embed(chunk.text)

Current model:

BAAI/bge-small-en-v1 (via sentence-transformers)

7. Store vectors

Each chunk becomes a row in the vector database.

---

# Vector Database

SmartFiles uses:

Chroma

This is a local vector database optimized for embedding similarity search.

Each stored chunk contains:

chunk_id  
embedding  
text  
filepath  
page_number  
metadata

Database location (configurable):

By default, SmartFiles stores its data under:

~/.smartfiles/

This base directory can be overridden with the environment variable
`SMARTFILES_DATA_DIR`. Within that directory, the vector database and
corpus are stored as:

- Vector DB: `<DATA_DIR>/database/`
- Text corpus: `<DATA_DIR>/corpus/`

In parallel with the vector database, SmartFiles maintains a plain
text corpus for validation and debugging:

- Location: `<DATA_DIR>/corpus/`
- Contents: one UTF-8 `.txt` file per indexed document containing the
	full raw extracted text (after parsing/OCR, before chunking).

This corpus is regenerated when running `smartfiles index` with
`--recreate`.

---

# Search Behavior

When a query is submitted:

1. Query is embedded using the same embedding model.

2. Vector database returns nearest neighbors.

3. Results are reranked using metadata signals.

Example ranking formula:

score =
0.7 * semantic_similarity
+ 0.2 * metadata_match
+ 0.1 * recency

Results are converted to a score from 0–100.

---

# User Interface

The UI should mimic a **Finder-style workflow**.

Layout:

Search bar

Results list | Preview pane

Users should be able to quickly skim documents with preview functionality similar to macOS Quick Look.

---

# Design Goals

SmartFiles should be:

• local-first  
• fast  
• simple architecture  
• modular  
• easy to install  

The project should avoid unnecessary frameworks and keep the pipeline understandable.

LangChain or heavy AI orchestration frameworks are intentionally avoided.

---

# Installation Goals

Primary install method:

pip install smartfiles

Future install options:

Homebrew

Eventually:

SmartFiles desktop app (Tauri or Electron)

---

# Initial Development Strategy

Development should start with a **thin vertical slice**:

1. CLI
2. Embedding generation
3. Vector storage
4. Basic search

Ignore for now:

OCR  
UI  
metadata tagging  
watch mode  

Those will be added after the core pipeline works.

---

# Long-Term Features

Planned future improvements:

• OCR for scanned PDFs  
• automatic document tagging  
• background indexing  
• Finder-style preview UI  
• desktop application  
• improved ranking algorithms  

---

# Summary

SmartFiles is a local semantic search tool for document folders.

It converts documents into embeddings and uses vector similarity search to retrieve the most relevant results.

The architecture is intentionally simple and modular to make development straightforward and maintainable.

