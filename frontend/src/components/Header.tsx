import type { FormEvent } from 'react'
import type { Folder, Health } from './types'

export type HeaderProps = {
  health: Health | null
  healthError: string | null
  query: string
  k: number
  isSearching: boolean
  searchError: string | null
  activeTab: 'search' | 'folders'
  onTabChange: (tab: 'search' | 'folders') => void
  folders: Folder[]
  foldersError: string | null
  selectedFolderNames: string[]
  onSelectedFolderNamesChange: (values: string[]) => void
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
  activeTab,
  onTabChange,
  folders,
  foldersError,
  selectedFolderNames,
  onSelectedFolderNamesChange,
  onQueryChange,
  onKChange,
  onSearchSubmit,
}: HeaderProps) {
  return (
    <header className="app-header">
      <div className="header-top">
        <div className="header-left">
          <h1>SmartFiles</h1>
          <div className="tab-bar">
            <button
              type="button"
              className={`tab-button${activeTab === 'search' ? ' tab-button-active' : ''}`}
              onClick={() => onTabChange('search')}
            >
              Search
            </button>
            <button
              type="button"
              className={`tab-button${activeTab === 'folders' ? ' tab-button-active' : ''}`}
              onClick={() => onTabChange('folders')}
            >
              Manage Folders
            </button>
          </div>
        </div>
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

      {activeTab === 'search' && (
        <>
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
                <p className="helper-text">No indexed folders yet. Use the Manage Folders tab to add one.</p>
              )}
              {folders.length > 0 && (
                <p className="helper-text">Leave unselected to search across all indexed folders.</p>
              )}
            </div>
          </div>
        </>
      )}
    </header>
  )
}
