# SmartFiles Architecture

SmartFiles is a local semantic search engine for document folders.

## High-Level Pipeline

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
User Interface

## Major Components

### Ingestion Pipeline
Processes documents into searchable chunks.

Steps:
1. Scan folders
2. Extract text
3. Chunk documents
4. Generate embeddings
5. Store in vector database

### Vector Database
Stores embeddings and metadata.

Initial implementation: Chroma

### Search Engine
Handles queries:
1. Embed query
2. Vector similarity search
3. Metadata-aware reranking
4. Return ranked documents

### API Server
Provides a FastAPI backend.

Endpoints:
POST /search
GET /document

### UI
Finder-style search interface:
- search bar
- ranked results
- preview pane
