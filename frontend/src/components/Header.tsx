import type { FormEvent } from 'react'
import type { Health } from './types'

export type HeaderProps = {
  health: Health | null
  healthError: string | null
  query: string
  k: number
  isSearching: boolean
  searchError: string | null
  onQueryChange: (value: string) => void
  onKChange: (value: number) => void
  onSearchSubmit: (event: FormEvent<HTMLFormElement>) => void
}

export default function Header({
  health,
  healthError,
  query,
  k,
  isSearching,
  searchError,
  onQueryChange,
  onKChange,
  onSearchSubmit,
}: HeaderProps) {
  return (
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

      <form className="search-bar" onSubmit={onSearchSubmit}>
        <label className="field">
          <span>Query</span>
          <input
            type="text"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
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
            onChange={(e) => onKChange(Number(e.target.value) || 1)}
          />
        </label>

        <button type="submit" className="primary-button" disabled={isSearching}>
          {isSearching ? 'Searching…' : 'Search'}
        </button>
      </form>
      {searchError && <p className="error-text error-inline">{searchError}</p>}
    </header>
  )
}
