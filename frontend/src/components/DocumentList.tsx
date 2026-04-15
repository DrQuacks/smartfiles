import type { KeyboardEvent } from 'react'
import { useEffect, useRef } from 'react'
import { API_BASE_URL } from '../config'
import type { SearchResult } from './types'
import type { AggregatedResult } from './searchUtils'
import { formatPageRange, getFileName } from './searchUtils'

export type DocumentListProps = {
  results: SearchResult[]
  selectedIndex: number | null
  onSelectedIndexChange: (index: number | null) => void
  searchError: string | null
  isSearching: boolean
}

export default function DocumentList({
  results,
  selectedIndex,
  onSelectedIndexChange,
  searchError,
  isSearching,
}: DocumentListProps) {
  const listRef = useRef<HTMLUListElement | null>(null)

  useEffect(() => {
    if (selectedIndex == null || !listRef.current) return
    const item = listRef.current.children[selectedIndex] as HTMLElement | undefined
    if (item) {
      item.scrollIntoView({ block: 'nearest' })
    }
  }, [selectedIndex])

  return (
    <section className="panel list-panel">
      <h2>Documents</h2>
      {results.length === 0 && !searchError && !isSearching && (
        <p className="muted">No results yet. Try a query above.</p>
      )}
      {results.length === 0 && isSearching && (
        <p className="helper-text small">Searching… fetching results and computing scores.</p>
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
        <ul className="results-list" ref={listRef}>
          {results.map((result, index) => {
            const pageSummary =
              (result as AggregatedResult).page_summary ?? formatPageRange(result)
            const isSelected = selectedIndex === index
            const fileName = getFileName(result.filepath) ?? '(unknown file)'
            const fileUrl = result.filepath
              ? `${API_BASE_URL}/file?filepath=${encodeURIComponent(result.filepath)}`
              : null
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
                  {pageSummary && (
                    <span className="result-page">{pageSummary}</span>
                  )}
                  {result.rerank_score == null ? (
                    <span className="result-score result-score-skeleton" />
                  ) : (
                    <span className="result-score">
                      {result.rerank_score.toFixed(1)}
                    </span>
                  )}
                </div>
                <div className="result-scores-row">
                  <span className="result-score-variant">
                    base: {result.score.toFixed(1)}
                  </span>
                  <span className="result-score-variant">
                    -20%: {result.score_drop20 != null ? result.score_drop20.toFixed(1) : '–'}
                  </span>
                  <span className="result-score-variant">
                    -40%: {result.score_drop40 != null ? result.score_drop40.toFixed(1) : '–'}
                  </span>
                  <span className="result-score-variant">
                    -60%: {result.score_drop60 != null ? result.score_drop60.toFixed(1) : '–'}
                  </span>
                  <span className="result-score-variant">
                    -80%: {result.score_drop80 != null ? result.score_drop80.toFixed(1) : '–'}
                  </span>
                </div>
                {result.filepath && (
                  <div className="result-meta">
                    {fileUrl ? (
                      <a
                        className="result-path result-path-link"
                        href={fileUrl}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {result.filepath}
                      </a>
                    ) : (
                      <span className="result-path">{result.filepath}</span>
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
