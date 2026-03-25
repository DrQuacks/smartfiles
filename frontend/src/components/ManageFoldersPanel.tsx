import type { FormEvent } from 'react'
import type { Folder } from './types'

export type ManageFoldersPanelProps = {
  folders: Folder[]
  foldersError: string | null
  indexPath: string
  isIndexing: boolean
  indexError: string | null
  indexStatus: string | null
  onIndexPathChange: (value: string) => void
  onIndexFolderSubmit: (event: FormEvent<HTMLFormElement>) => void
}

export default function ManageFoldersPanel({
  folders,
  foldersError,
  indexPath,
  isIndexing,
  indexError,
  indexStatus,
  onIndexPathChange,
  onIndexFolderSubmit,
}: ManageFoldersPanelProps) {
  return (
    <section className="panel manage-folders-panel">
      <h2>Manage Folders</h2>

      <div className="manage-folders-content">
        <div className="folder-list-section">
          <h3>Indexed Folders</h3>
          {foldersError && <p className="error-text">{foldersError}</p>}
          {!foldersError && folders.length === 0 && (
            <p className="muted">No folders have been indexed yet.</p>
          )}
          {folders.length > 0 && (
            <ul className="folder-list">
              {folders.map((folder) => (
                <li key={folder.folder_name} className="folder-item">
                  <div className="folder-item-main">
                    <span className="folder-name">{folder.folder_name}</span>
                    <span className="folder-path">{folder.path}</span>
                  </div>
                  {folder.last_indexed && (
                    <span className="folder-meta">Last indexed: {folder.last_indexed}</span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>

        <form className="index-form" onSubmit={onIndexFolderSubmit}>
          <h3>Index New Folder</h3>
          <label className="field">
            <span>Folder path</span>
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
    </section>
  )
}
