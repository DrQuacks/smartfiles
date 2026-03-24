import type { FormEvent } from 'react'
import type { Folder, Health } from './types'

export type HeaderProps = {
  health: Health | null
  healthError: string | null
  query: string
  k: number
  isSearching: boolean
  searchError: string | null
  folders: Folder[]
  foldersError: string | null
  selectedFolderNames: string[]
  onSelectedFolderNamesChange: (values: string[]) => void
  indexPath: string
  isIndexing: boolean
  indexError: string | null
  indexStatus: string | null
  onIndexPathChange: (value: string) => void
  onQueryChange: (value: string) => void
  onKChange: (value: number) => void
  onSearchSubmit: (event: FormEvent<HTMLFormElement>) => void
  onIndexFolderSubmit: (event: FormEvent<HTMLFormElement>) => void
}

export default function Header({
  health,
  healthError,
  query,
  k,
  isSearching,
  searchError,
  folders,
  foldersError,
  selectedFolderNames,
  onSelectedFolderNamesChange,
  indexPath,
  isIndexing,
  indexError,
  indexStatus,
  onIndexPathChange,
  onQueryChange,
  onKChange,
  onSearchSubmit,
  onIndexFolderSubmit,
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

      <div className="folder-controls">
        <div className="folder-scope">
          <label className="field">
            <span>Search folders</span>
            <select
              multiple
              value={selectedFolderNames}
              onChange={(e) => {
                const values = Array.from(e.target.selectedOptions).map((opt) => opt.value)
                onSelectedFolderNamesChange(values)
              }}
            >
              {folders.map((folder) => (
                <option key={folder.folder_name} value={folder.folder_name}>
                  {folder.folder_name}
                </option>
              ))}
            </select>
          </label>
          {foldersError && <p className="error-text error-inline">{foldersError}</p>}
          {!foldersError && folders.length === 0 && (
            <p className="helper-text">No indexed folders yet. Use the form below to index one.</p>
          )}
          {folders.length > 0 && (
            <p className="helper-text">Leave unselected to search across all indexed folders.</p>
          )}
        </div>

        <form className="index-form" onSubmit={onIndexFolderSubmit}>
          <label className="field">
            <span>Index new folder</span>
            <input
              type="text"
              value={indexPath}
              onChange={(e) => onIndexPathChange(e.target.value)}
              placeholder="/absolute/path/to/folder"
            />
          </label>
          <button type="submit" className="secondary-button" disabled={isIndexing}>
            {isIndexing ? 'Indexing…' : 'Run index pipeline'}
          </button>
          {indexError && <p className="error-text error-inline">{indexError}</p>}
          {indexStatus && !indexError && <p className="success-text">{indexStatus}</p>}
        </form>
      </div>
    </header>
  )
}
