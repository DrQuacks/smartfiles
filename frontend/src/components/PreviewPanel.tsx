import type { SearchResult } from './types'
import { getFileExtension, getFileName } from './searchUtils'
import { API_BASE_URL } from '../config'
import DocxPreview from './DocxPreview'

function buildFileUrl(result: SearchResult): string | null {
  if (!result.filepath) return null
  const base = `${API_BASE_URL}/file?filepath=${encodeURIComponent(result.filepath)}`
  const ext = getFileExtension(result.filepath)
  if (ext === 'pdf') {
    const pageFragment = result.page_start != null ? `page=${result.page_start}` : ''
    const viewerParams = ['toolbar=0', 'navpanes=0', 'scrollbar=1']
    const fragmentParts = [pageFragment, ...viewerParams].filter(Boolean)
    return fragmentParts.length ? `${base}#${fragmentParts.join('&')}` : base
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
      <div className="preview-frame preview-image-frame">
        <img
          className="preview-image"
          src={fileUrl}
          alt="Document preview"
        />
      </div>
    )
  }

  if (ext === 'docx') {
    return (
      <div className="preview-frame preview-docx-frame">
        <DocxPreview fileUrl={fileUrl} />
      </div>
    )
  }

  return (
    <div className="preview-frame preview-fallback">
      <p className="preview-fallback-text">
        Inline preview is not available for this file type.
      </p>
      <a
        className="preview-link"
        href={fileUrl}
        target="_blank"
        rel="noreferrer"
      >
        Open file in new tab
      </a>
    </div>
  )
}

export type PreviewPanelProps = {
  selectedResult: SearchResult | null
}

export default function PreviewPanel({ selectedResult }: PreviewPanelProps) {
  return (
    <section className="panel preview-panel">
      <h2>Preview</h2>
      {selectedResult ? (
        <div className="preview-container">
          {selectedResult.filepath && (
            <>
              <p className="preview-name">
                {getFileName(selectedResult.filepath)}
              </p>
            </>
          )}
          {renderPreview(selectedResult)}
        </div>
      ) : (
        <p className="muted">Run a search and select a document.</p>
      )}
    </section>
  )
}
