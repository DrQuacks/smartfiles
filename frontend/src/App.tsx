import { useEffect, useState } from 'react'
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
    const trimmed = query.trim()
    if (!trimmed) {
      setResults([])
      setSearchError(null)
      setSelectedIndex(null)
      return
    }

    setIsSearching(true)
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
      setResults(aggregated)
      setSelectedIndex(aggregated.length > 0 ? 0 : null)
    } catch (error) {
      console.error(error)
      setSearchError('Search request failed')
      setResults([])
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
