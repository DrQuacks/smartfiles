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
}
