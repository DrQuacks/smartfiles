# SmartFiles Indexing Pipeline

The indexing pipeline converts documents into searchable vector embeddings.

## Steps

### 1. File Scanning
Detect new or modified files.

Initial supported types (implemented):
- PDF
- PNG
- JPG

Planned (not yet implemented):
- DOCX

### 2. Text Extraction
PDF parsing using pypdf.

PNG / JPG:
- OCR using pytesseract (via Pillow image loading)

If OCR dependencies are not installed, image files are currently skipped (no text extracted).

### 3. Text Cleaning
Normalize text:
- remove repeated headers
- normalize whitespace

### 4. Chunking

Current implementation uses simple word-based chunks as an approximation of tokens:

- chunk_size ≈ 500 words
- overlap ≈ 50 words

Small documents may use page-level chunks.

### 5. Metadata Extraction

Possible tags:
school
course
topic

### 6. Embedding

Example model:
bge-small-en

### 7. Storage

Embeddings and metadata stored in vector database.
