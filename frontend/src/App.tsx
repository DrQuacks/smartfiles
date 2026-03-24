import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'
import { API_BASE_URL } from './config'
import type { Health, SearchResult } from './components/types'
import Header from './components/Header'
import DocumentList from './components/DocumentList'
import PreviewPanel from './components/PreviewPanel'
import { dedupeResultsByFile } from './components/searchUtils'

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  const [query, setQuery] = useState('')
  const [k, setK] = useState(5)
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
      const chunksK = Math.min(Math.max(k * 5, k + 5), 100)
      const params = new URLSearchParams({
        query: trimmed,
        k: chunksK.toString(),
      })
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

  return (
    <div className="app-container">
      <Header
        health={health}
        healthError={healthError}
        query={query}
        k={k}
        isSearching={isSearching}
        searchError={searchError}
        onQueryChange={setQuery}
        onKChange={setK}
        onSearchSubmit={handleSearch}
      />

      <main className="app-main">
        <DocumentList
          results={results}
          selectedIndex={selectedIndex}
          onSelectedIndexChange={setSelectedIndex}
          searchError={searchError}
        />

        <PreviewPanel selectedResult={selectedResult} />
      </main>
    </div>
  )
}

export default App
