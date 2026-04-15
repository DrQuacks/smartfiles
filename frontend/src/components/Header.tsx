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
  const handleAddFolderFilter = (value: string) => {
    const trimmed = value.trim()
    if (!trimmed) return
    if (selectedFolderNames.includes(trimmed)) return
    onSelectedFolderNamesChange([...selectedFolderNames, trimmed])
  }

  const handleRemoveFolderFilter = (name: string) => {
    onSelectedFolderNamesChange(selectedFolderNames.filter((n) => n !== name))
  }

  const isSearchDisabled =
    isSearching || !query.trim() || selectedFolderNames.length === 0

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

            <label className="field field-folder-select">
              <span>Search folders</span>
              <select
                onChange={(e) => {
                  const value = e.target.value
                  if (!value) return
                  handleAddFolderFilter(value)
                }}
                defaultValue=""
              >
                <option value="" disabled>
                  {folders.length === 0 ? 'No indexed folders' : 'All indexed folders'}
                </option>
                {folders.map((folder) => (
                  <option key={folder.folder_name} value={folder.folder_name}>
                    {folder.folder_name}
                  </option>
                ))}
              </select>
            </label>

            <button type="submit" className="primary-button" disabled={isSearchDisabled}>
              {isSearching ? 'Searching…' : 'Search'}
            </button>
          </form>
          {searchError && <p className="error-text error-inline">{searchError}</p>}

          {foldersError && folders.length === 0 && (
            <p className="error-text error-inline">{foldersError}</p>
          )}
          {isSearching && !searchError && (
            <p className="helper-text small search-progress">
              Running search and computing similarity variants… this can take a few seconds.
            </p>
          )}
          {!foldersError && folders.length === 0 && (
            <p className="helper-text">No indexed folders yet. Use the Manage Folders tab to add one.</p>
          )}
          {!foldersError && folders.length > 0 && selectedFolderNames.length === 0 && (
            <p className="helper-text">Leave empty to search across all indexed folders.</p>
          )}
          {selectedFolderNames.length > 0 && (
            <div className="folder-chip-row">
              {selectedFolderNames.map((name) => (
                <button
                  key={name}
                  type="button"
                  className="folder-chip"
                  onClick={() => handleRemoveFolderFilter(name)}
                >
                  <span className="folder-chip-label">{name}</span>
                  <span className="folder-chip-remove" aria-hidden="true">
                    ×
                  </span>
                </button>
              ))}
            </div>
          )}
        </>
      )}
    </header>
  )
}
