# SmartFiles Indexing Pipeline

The indexing pipeline converts documents into searchable vector embeddings.

## Steps

### 1. File Scanning
Detect new or modified files.

Supported types:
- PDF
- DOCX
- PNG
- JPG

### 2. Text Extraction
PDF parsing using pypdf.

OCR fallback using pytesseract.

### 3. Text Cleaning
Normalize text:
- remove repeated headers
- normalize whitespace

### 4. Chunking

chunk_size = 500 tokens
overlap = 50 tokens

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
