import { useEffect, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
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

function getFileName(path?: string): string | null {
  if (!path) return null
  const parts = path.split(/[\\/]/)
  return parts[parts.length - 1] ?? null
}

function getFileExtension(path?: string): string | null {
  if (!path) return null
  const lastDot = path.lastIndexOf('.')
  if (lastDot === -1) return null
  return path.slice(lastDot + 1).toLowerCase()
}

function buildFileUrl(result: SearchResult): string | null {
  if (!result.filepath) return null
  const base = `${API_BASE_URL}/file?filepath=${encodeURIComponent(result.filepath)}`
  const ext = getFileExtension(result.filepath)
  if (ext === 'pdf' && result.page_start != null) {
    return `${base}#page=${result.page_start}`
  }
  return base
}

function renderPreview(result: SearchResult) {
  const fileUrl = buildFileUrl(result)
  const ext = getFileExtension(result.filepath)

  if (!fileUrl) {
    return <p className="muted">No file path available for this result.</p>
  }

  if (ext === 'pdf') {
    return (
      <iframe
        className="preview-frame"
        src={fileUrl}
        title="PDF preview"
      />
    )
  }

  if (ext === 'png' || ext === 'jpg' || ext === 'jpeg') {
    return (
      <img
        className="preview-image"
        src={fileUrl}
        alt="Document preview"
      />
    )
  }

  return (
    <a
      className="preview-link"
      href={fileUrl}
      target="_blank"
      rel="noreferrer"
    >
      Open file in new tab
    </a>
  )
}

function formatPageRange(result: SearchResult): string | null {
  const { page_start, page_end } = result
  if (page_start == null && page_end == null) return null
  if (page_start != null && page_end != null) {
    if (page_start === page_end) return `Page ${page_start}`
    return `Pages ${page_start}–${page_end}`
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
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const selectedResult = selectedIndex != null ? results[selectedIndex] : null
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

  const handleSearch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const trimmed = query.trim()
    if (!trimmed) {
      setResults([])
      setSearchError(null)
      setSelectedIndex(null)
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
      setSelectedIndex(data.length > 0 ? 0 : null)
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
        <div className="header-top">
          <h1>SmartFiles</h1>
          <div className="status-bar">
            {health ? (
              <span className="status-pill status-ok">Backend OK</span>
            ) : healthError ? (
              <span className="status-pill status-error">Backend offline</span>
            ) : (
              <span className="status-pill">Checking backend…</span>
            )}
            {health && (
              <span className="status-detail">
                Model: {health.model_loaded ? 'loaded' : 'not loaded'} · Index:{' '}
                {health.vector_store_initialized ? 'ready' : 'not ready'}
              </span>
            )}
          </div>
        </div>

        <form className="search-bar" onSubmit={handleSearch}>
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
        {searchError && <p className="error-text error-inline">{searchError}</p>}
      </header>

      <main className="app-main">
        <section className="panel list-panel">
          <h2>Documents</h2>
          {results.length === 0 && !searchError && (
            <p className="muted">No results yet. Try a query above.</p>
          )}

          <div
            className="document-list"
            tabIndex={0}
            onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
              if (results.length === 0) return

              if (event.key === 'ArrowDown') {
                event.preventDefault()
                setSelectedIndex((current) => {
                  if (current == null) return 0
                  return Math.min(current + 1, results.length - 1)
                })
              } else if (event.key === 'ArrowUp') {
                event.preventDefault()
                setSelectedIndex((current) => {
                  if (current == null) return results.length - 1
                  return Math.max(current - 1, 0)
                })
              }
            }}
          >
            <ul className="results-list">
              {results.map((result, index) => {
                const pageLabel = formatPageRange(result)
                const isSelected = selectedIndex === index
                const fileName = getFileName(result.filepath) ?? '(unknown file)'
                return (
                  <li
                    key={result.id}
                    className={`result-item${
                      isSelected ? ' result-item-selected' : ''
                    }`}
                    onClick={() => setSelectedIndex(index)}
                  >
                    <div className="result-header">
                      <span className="result-name">{fileName}</span>
                      {pageLabel && (
                        <span className="result-page">{pageLabel}</span>
                      )}
                      <span className="result-score">
                        {result.score.toFixed(1)}
                      </span>
                    </div>
                    {result.filepath && (
                      <div className="result-meta">
                        <span className="result-path">{result.filepath}</span>
                      </div>
                    )}
                  </li>
                )
              })}
            </ul>
          </div>
        </section>

        <section className="panel preview-panel">
          <h2>Preview</h2>
          {selectedResult ? (
            <div className="preview-container">
              {selectedResult.filepath && (
                <>
                  <p className="preview-name">
                    {getFileName(selectedResult.filepath)}
                  </p>
                  <p className="preview-path">{selectedResult.filepath}</p>
                </>
              )}
              {renderPreview(selectedResult)}
            </div>
          ) : (
            <p className="muted">Run a search and select a document.</p>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
