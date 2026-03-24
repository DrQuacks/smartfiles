import type { KeyboardEvent } from 'react'
import type { SearchResult } from './types'
import type { AggregatedResult } from './searchUtils'
import { formatPageRange, getFileName } from './searchUtils'

export type DocumentListProps = {
  results: SearchResult[]
  selectedIndex: number | null
  onSelectedIndexChange: (index: number | null) => void
  searchError: string | null
}

export default function DocumentList({
  results,
  selectedIndex,
  onSelectedIndexChange,
  searchError,
}: DocumentListProps) {
  return (
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
            onSelectedIndexChange(
              selectedIndex == null
                ? 0
                : Math.min(selectedIndex + 1, results.length - 1),
            )
          } else if (event.key === 'ArrowUp') {
            event.preventDefault()
            onSelectedIndexChange(
              selectedIndex == null
                ? results.length - 1
                : Math.max(selectedIndex - 1, 0),
            )
          }
        }}
      >
        <ul className="results-list">
          {results.map((result, index) => {
            const hitPages = (result as AggregatedResult).hit_pages
            const pageLabels =
              hitPages && hitPages.length > 0
                ? hitPages
                : (() => {
                    const label = formatPageRange(result)
                    return label ? [label] : []
                  })()
            const primaryPage = pageLabels[0]
            const isSelected = selectedIndex === index
            const fileName = getFileName(result.filepath) ?? '(unknown file)'
            return (
              <li
                key={result.id}
                className={`result-item${
                  isSelected ? ' result-item-selected' : ''
                }`}
                onClick={() => onSelectedIndexChange(index)}
              >
                <div className="result-header">
                  <span className="result-name">{fileName}</span>
                  {primaryPage && (
                    <span className="result-page">{primaryPage}</span>
                  )}
                  <span className="result-score">
                    {result.score.toFixed(1)}
                  </span>
                </div>
                {result.filepath && (
                  <div className="result-meta">
                    <span className="result-path">{result.filepath}</span>
                    {pageLabels.length > 1 && (
                      <div className="result-pages-list">
                        {pageLabels.map((label) => (
                          <span
                            key={label}
                            className="result-page-chip"
                          >
                            {label}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </li>
            )
          })}
        </ul>
      </div>
    </section>
  )
}
