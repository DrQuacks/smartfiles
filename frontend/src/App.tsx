import { useEffect, useState } from 'react'
import './App.css'

type Health = {
  status: string
  model_loaded: boolean
  vector_store_initialized: boolean
}

type SearchResult = {
  id: string
  text: string
  score: number
  filepath?: string
  chunk_index?: number
  page_start?: number
  page_end?: number
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

function formatPageRange(result: SearchResult): string | null {
  const { page_start, page_end } = result
  if (page_start == null && page_end == null) return null
  if (page_start != null && page_end != null) {
    if (page_start === page_end) return `Page ${page_start}`
    return `Pages ${page_start}${page_end}`
  }
  if (page_start != null) return `From page ${page_start}`
  if (page_end != null) return `Up to page ${page_end}`
  return null
}

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  const [query, setQuery] = useState('')
  const [k, setK] = useState(5)
  const [results, setResults] = useState<SearchResult[]>([])
  const [searchError, setSearchError] = useState<string | null>(null)
  const [isSearching, setIsSearching] = useState(false)

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        setHealthError(null)
        const res = await fetch(`${API_BASE_URL}/health`)
        if (!res.ok) {
          throw new Error(`Health check failed: ${res.status}`)
        }
        const data: Health = await res.json()
        setHealth(data)
      } catch (error) {
        console.error(error)
        setHealth(null)
        setHealthError('Unable to reach backend')
      }
    }

    fetchHealth()
  }, [])

  const handleSearch = async (event: React.FormEvent) => {
    event.preventDefault()
    const trimmed = query.trim()
    if (!trimmed) {
      setResults([])
      setSearchError(null)
      return
    }

    setIsSearching(true)
    setSearchError(null)

    try {
      const params = new URLSearchParams({ query: trimmed, k: k.toString() })
      const res = await fetch(`${API_BASE_URL}/search?${params.toString()}`)
      if (!res.ok) {
        throw new Error(`Search failed: ${res.status}`)
      }
      const data: SearchResult[] = await res.json()
      setResults(data)
    } catch (error) {
      console.error(error)
      setSearchError('Search request failed')
      setResults([])
    } finally {
      setIsSearching(false)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>SmartFiles</h1>
        <div className="status-bar">
          {health ? (
            <span className="status-pill status-ok">Backend OK</span>
          ) : healthError ? (
            <span className="status-pill status-error">Backend offline</span>
          ) : (
            <span className="status-pill">Checking backend3</span>
          )}
          {health && (
            <span className="status-detail">
              Model: {health.model_loaded ? 'loaded' : 'not loaded'} · Index:{' '}
              {health.vector_store_initialized ? 'ready' : 'not ready'}
            </span>
          )}
        </div>
      </header>

      <main className="app-main">
        <section className="panel search-panel">
          <h2>Search your documents</h2>
          <form className="search-form" onSubmit={handleSearch}>
            <label className="field">
              <span>Query</span>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. gradient descent on page 3"
              />
            </label>

            <label className="field field-inline">
              <span>Top K</span>
              <input
                type="number"
                min={1}
                max={50}
                value={k}
                onChange={(e) => setK(Number(e.target.value) || 1)}
              />
            </label>

            <button type="submit" className="primary-button" disabled={isSearching}>
              {isSearching ? 'Searching…' : 'Search'}
            </button>
          </form>
          {searchError && <p className="error-text">{searchError}</p>}
        </section>

        <section className="panel results-panel">
          <h2>Results</h2>
          {results.length === 0 && !searchError && (
            <p className="muted">No results yet. Try a query above.</p>
          )}

          <ul className="results-list">
            {results.map((result) => {
              const pageLabel = formatPageRange(result)
              return (
                <li key={result.id} className="result-item">
                  <div className="result-header">
                    <span className="result-score">
                      {result.score.toFixed(1)}
                    </span>
                    {result.filepath && (
                      <span className="result-path">{result.filepath}</span>
                    )}
                    {pageLabel && (
                      <span className="result-page">{pageLabel}</span>
                    )}
                  </div>
                  <p className="result-text">{result.text}</p>
                </li>
              )
            })}
          </ul>
        </section>
      </main>
    </div>
  )
}

export default App
