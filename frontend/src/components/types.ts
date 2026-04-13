export type Health = {
  status: string
  model_loaded: boolean
  vector_store_initialized: boolean
}

export type SearchResult = {
  id: string
  text: string
  score: number
  filepath?: string
  chunk_index?: number
  page_start?: number
  page_end?: number
  // Optional score from a secondary reranker stage. When present,
  // the UI prefers this over the base `score` from vector search.
  rerank_score?: number | null
}

export type Folder = {
  folder_name: string
  path: string
  raw_text_dir_name: string
  last_indexed?: string | null
  last_commit?: string | null
}
