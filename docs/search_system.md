# SmartFiles Search System

The search system retrieves documents using semantic similarity.

## Query Flow

User Query
 ↓
Query Embedding
 ↓
Vector Search
 ↓
Reranking
 ↓
Results

## Query Embedding

The query is converted into an embedding using the same model used during indexing.

## Vector Search

Vector database returns most similar chunks.

## Reranking

Example scoring:

score =
0.7 * semantic_similarity
+ 0.2 * metadata_match
+ 0.1 * recency

Metadata signals:
- filename
- detected course
- school name
