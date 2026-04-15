import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'
import { API_BASE_URL } from './config'
import type { Folder, Health, SearchResult } from './components/types'
import Header from './components/Header'
import DocumentList from './components/DocumentList'
import PreviewPanel from './components/PreviewPanel'
import ManageFoldersPanel from './components/ManageFoldersPanel'
import { dedupeResultsByFile } from './components/searchUtils'

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  const [query, setQuery] = useState('')
  const [k, setK] = useState(5)
  const [folders, setFolders] = useState<Folder[]>([])
  const [selectedFolderNames, setSelectedFolderNames] = useState<string[]>([])
  const [foldersError, setFoldersError] = useState<string | null>(null)

  const [indexPath, setIndexPath] = useState('')
  const [isIndexing, setIsIndexing] = useState(false)
  const [indexError, setIndexError] = useState<string | null>(null)
  const [indexStatus, setIndexStatus] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'search' | 'folders'>('search')
  const [results, setResults] = useState<SearchResult[]>([])
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const selectedResult = selectedIndex != null ? results[selectedIndex] : null
  const [searchError, setSearchError] = useState<string | null>(null)
  const [isSearching, setIsSearching] = useState(false)
  const [isDimdropScoring, setIsDimdropScoring] = useState(false)
  const [dimdropCompletedSteps, setDimdropCompletedSteps] = useState(0)
  const dimdropTotalSteps = 4
  const searchRunRef = useRef(0)

  type DropField = 'score_drop50' | 'score_drop75' | 'score_drop90' | 'score_drop95'

  const applyDropScore = (
    result: SearchResult,
    field: DropField,
    value: number,
  ): SearchResult => {
    if (field === 'score_drop50') return { ...result, score_drop50: value }
    if (field === 'score_drop75') return { ...result, score_drop75: value }
    if (field === 'score_drop90') return { ...result, score_drop90: value }
    return { ...result, score_drop95: value }
  }

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

  useEffect(() => {
    const fetchFolders = async () => {
      try {
        setFoldersError(null)
        const res = await fetch(`${API_BASE_URL}/folders`)
        if (!res.ok) {
          throw new Error(`Folders fetch failed: ${res.status}`)
        }
        const data: Folder[] = await res.json()
        setFolders(data)
      } catch (error) {
        console.error(error)
        setFolders([])
        setFoldersError('Unable to load indexed folders')
      }
    }

    fetchFolders()
  }, [])

  const handleSearch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const runId = searchRunRef.current + 1
    searchRunRef.current = runId

    const trimmed = query.trim()
    if (!trimmed) {
      setResults([])
      setSearchError(null)
      setSelectedIndex(null)
      setIsDimdropScoring(false)
      setDimdropCompletedSteps(0)
      return
    }

    setIsSearching(true)
    setIsDimdropScoring(false)
    setDimdropCompletedSteps(0)
    setSearchError(null)

    try {
      // Over-fetch chunks so we have enough unique documents after
      // collapsing multiple hits from the same file.
      const chunksK = Math.min(Math.max(k * 10, k + 10), 400)
      const params = new URLSearchParams({
        query: trimmed,
        k: chunksK.toString(),
      })
      if (selectedFolderNames.length > 0) {
        params.set('folders', selectedFolderNames.join(','))
      }
      const res = await fetch(`${API_BASE_URL}/search?${params.toString()}`)
      if (!res.ok) {
        throw new Error(`Search failed: ${res.status}`)
      }
      const data: SearchResult[] = await res.json()
      const aggregated = dedupeResultsByFile(data, k)

      // Show initial results immediately with pending scores; the UI
      // will render skeleton pills until rerank scores arrive.
      const withPending = aggregated.map((r) => ({
        ...r,
        rerank_score: null,
        score_drop50: null,
        score_drop75: null,
        score_drop90: null,
        score_drop95: null,
      }))
      setResults(withPending)
      setSelectedIndex(withPending.length > 0 ? 0 : null)

      // Fire-and-forget rerank request; when it returns we hydrate
      // rerank_score values and optionally adjust ordering.
      void (async () => {
        try {
          const rerankRes = await fetch(`${API_BASE_URL}/search/rerank`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              query: trimmed,
              results: withPending.map((r) => ({
                id: r.id,
                text: r.text,
                score: r.score,
                filepath: r.filepath,
              })),
            }),
          })
          if (!rerankRes.ok) {
            return
          }
          const reranked: { id: string; rerank_score: number }[] =
            await rerankRes.json()
          if (runId !== searchRunRef.current) return
          const scoreById = new Map(reranked.map((r) => [r.id, r.rerank_score]))

          setResults((current) => {
            const updated = current.map((r) => {
              const s = scoreById.get(r.id)
              if (s == null) return r
              return { ...r, rerank_score: s }
            })

            // Reorder by rerank_score when available, falling back
            // to original order for items that lack a rerank score.
            const sorted = [...updated].sort((a, b) => {
              const sa = a.rerank_score
              const sb = b.rerank_score
              if (sa == null && sb == null) return 0
              if (sa == null) return 1
              if (sb == null) return -1
              return sb - sa
            })
            return sorted
          })
        } catch (error) {
          console.error(error)
        }
      })()

      // Compute dim-drop scores progressively for the already selected
      // results list (20 -> 40 -> 60 -> 80).
      void (async () => {
        const variants: Array<{ fraction: number; field: DropField }> = [
          { fraction: 0.5, field: 'score_drop50' },
          { fraction: 0.75, field: 'score_drop75' },
          { fraction: 0.9, field: 'score_drop90' },
          { fraction: 0.95, field: 'score_drop95' },
        ]

        setIsDimdropScoring(true)
        setDimdropCompletedSteps(0)

        try {
          for (let i = 0; i < variants.length; i += 1) {
            if (runId !== searchRunRef.current) return
            const variant = variants[i]

            const dimdropRes = await fetch(`${API_BASE_URL}/search/dimdrop`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                query: trimmed,
                drop_fraction: variant.fraction,
                results: withPending.map((r) => ({
                  id: r.id,
                  text: r.text,
                  score: r.score,
                  filepath: r.filepath,
                })),
              }),
            })

            if (!dimdropRes.ok) {
              continue
            }

            const scored: { id: string; score: number }[] = await dimdropRes.json()
            if (runId !== searchRunRef.current) return

            const scoreById = new Map(scored.map((r) => [r.id, r.score]))
            setResults((current) =>
              current.map((r) => {
                const s = scoreById.get(r.id)
                if (s == null) return r
                return applyDropScore(r, variant.field, s)
              }),
            )
            setDimdropCompletedSteps(i + 1)
          }
        } catch (error) {
          console.error(error)
        } finally {
          if (runId === searchRunRef.current) {
            setIsDimdropScoring(false)
          }
        }
      })()
    } catch (error) {
      console.error(error)
      setSearchError('Search request failed')
      setResults([])
      setIsDimdropScoring(false)
      setDimdropCompletedSteps(0)
    } finally {
      setIsSearching(false)
    }
  }

  const handleReindexFolder = async (folderPath: string) => {
      const trimmed = folderPath.trim()
      if (!trimmed) {
        setIndexError('Folder path is missing for this entry')
        setIndexStatus(null)
        return
      }

      setIsIndexing(true)
      setIndexError(null)
      setIndexStatus(null)
      setIndexPath(trimmed)

      try {
        const res = await fetch(`${API_BASE_URL}/index`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ root_folder: trimmed, recreate: true }),
        })
        if (!res.ok) {
          throw new Error(`Re-indexing failed: ${res.status}`)
        }

        setIndexStatus('Re-indexing completed successfully')

      try {
        const foldersRes = await fetch(`${API_BASE_URL}/folders`)
        if (foldersRes.ok) {
          const data: Folder[] = await foldersRes.json()
          setFoldersError(null)
          setFolders(data)
        }
      } catch (error) {
        console.error(error)
      }
      } catch (error) {
        console.error(error)
        setIndexError('Re-indexing request failed')
      } finally {
        setIsIndexing(false)
      }
    }

  const handleIndexFolder = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const trimmed = indexPath.trim()
    if (!trimmed) {
      setIndexError('Please enter a folder path')
      setIndexStatus(null)
      return
    }

    setIsIndexing(true)
    setIndexError(null)
    setIndexStatus(null)

    try {
      const res = await fetch(`${API_BASE_URL}/index`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ root_folder: trimmed }),
      })
      if (!res.ok) {
        throw new Error(`Indexing failed: ${res.status}`)
      }

      setIndexStatus('Indexing completed successfully')

      // Refresh the list of indexed folders so the new one appears
      try {
        const foldersRes = await fetch(`${API_BASE_URL}/folders`)
        if (foldersRes.ok) {
          const data: Folder[] = await foldersRes.json()
          setFoldersError(null)
          setFolders(data)
        }
      } catch (error) {
        console.error(error)
      }
    } catch (error) {
      console.error(error)
      setIndexError('Indexing request failed')
    } finally {
      setIsIndexing(false)
    }
  }

  const handleDeleteFolder = async (folderName: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/folders/${encodeURIComponent(folderName)}`, {
        method: 'DELETE',
      })
      if (!res.ok) {
        throw new Error(`Delete failed: ${res.status}`)
      }

      // Refresh folders after deletion
      const foldersRes = await fetch(`${API_BASE_URL}/folders`)
      if (foldersRes.ok) {
        const data: Folder[] = await foldersRes.json()
        setFoldersError(null)
        setFolders(data)
      }
    } catch (error) {
      console.error(error)
      setFoldersError('Unable to delete folder')
    }
  }

  const handleReorderFolders = async (orderedFolderNames: string[]) => {
    try {
      const res = await fetch(`${API_BASE_URL}/folders/reorder`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ order: orderedFolderNames }),
      })
      if (!res.ok) {
        throw new Error(`Reorder failed: ${res.status}`)
      }
      const data: Folder[] = await res.json()
      setFoldersError(null)
      setFolders(data)
    } catch (error) {
      console.error(error)
      setFoldersError('Unable to reorder folders')
    }
  }

  return (
    <div className="app-container">
      <Header
        health={health}
        healthError={healthError}
        query={query}
        k={k}
        isSearching={isSearching}
        isDimdropScoring={isDimdropScoring}
        dimdropCompletedSteps={dimdropCompletedSteps}
        dimdropTotalSteps={dimdropTotalSteps}
        searchError={searchError}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        folders={folders}
        foldersError={foldersError}
        selectedFolderNames={selectedFolderNames}
        onSelectedFolderNamesChange={setSelectedFolderNames}
        onQueryChange={setQuery}
        onKChange={setK}
        onSearchSubmit={handleSearch}
      />

      {activeTab === 'search' ? (
        <main className="app-main">
          <DocumentList
            results={results}
            selectedIndex={selectedIndex}
            onSelectedIndexChange={setSelectedIndex}
            searchError={searchError}
            isSearching={isSearching}
          />

          <PreviewPanel selectedResult={selectedResult} />
        </main>
      ) : (
        <main className="app-main app-main-manage">
          <ManageFoldersPanel
            folders={folders}
            foldersError={foldersError}
            indexPath={indexPath}
            isIndexing={isIndexing}
            indexError={indexError}
            indexStatus={indexStatus}
            onIndexPathChange={setIndexPath}
            onIndexFolderSubmit={handleIndexFolder}
            onReindexFolder={handleReindexFolder}
            onDeleteFolder={handleDeleteFolder}
            onReorderFolders={handleReorderFolders}
          />
        </main>
      )}
    </div>
  )
}

export default App
