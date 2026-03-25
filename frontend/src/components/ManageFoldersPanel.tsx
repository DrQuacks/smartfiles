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
  onDeleteFolder: (folderName: string) => void
  onReorderFolders: (orderedFolderNames: string[]) => void
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
  onDeleteFolder,
  onReorderFolders,
}: ManageFoldersPanelProps) {
  const handleMove = (folderName: string, direction: 'up' | 'down') => {
    const names = folders.map((f) => f.folder_name)
    const index = names.indexOf(folderName)
    if (index === -1) return
    if (direction === 'up' && index === 0) return
    if (direction === 'down' && index === names.length - 1) return

    const newNames = [...names]
    const swapWith = direction === 'up' ? index - 1 : index + 1
    const tmp = newNames[swapWith]
    newNames[swapWith] = newNames[index]
    newNames[index] = tmp
    onReorderFolders(newNames)
  }

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
                    <div className="folder-item-header">
                      <span className="folder-name">{folder.folder_name}</span>
                      <div className="folder-actions">
                        <button
                          type="button"
                          className="icon-button"
                          onClick={() => handleMove(folder.folder_name, 'up')}
                          aria-label="Move folder up"
                        >
                          ↑
                        </button>
                        <button
                          type="button"
                          className="icon-button"
                          onClick={() => handleMove(folder.folder_name, 'down')}
                          aria-label="Move folder down"
                        >
                          ↓
                        </button>
                        <button
                          type="button"
                          className="text-button text-button-danger"
                          onClick={() => onDeleteFolder(folder.folder_name)}
                        >
                          Delete
                        </button>
                      </div>
                    </div>
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
          <h3>Add &amp; Index Folder</h3>
          <label className="field">
            <span>Folder path</span>
            <input
              type="text"
              value={indexPath}
              onChange={(e) => onIndexPathChange(e.target.value)}
              placeholder="/absolute/path/to/folder"
            />
          </label>
          <p className="helper-text small">
            Enter the full folder path on this machine. A native folder picker
            will be added in a future desktop version.
          </p>
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
