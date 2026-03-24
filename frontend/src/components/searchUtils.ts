import type { SearchResult } from './types'

export function getFileName(path?: string): string | null {
  if (!path) return null
  const parts = path.split(/[\\/]/)
  return parts[parts.length - 1] ?? null
}

export function getFileExtension(path?: string): string | null {
  if (!path) return null
  const lastDot = path.lastIndexOf('.')
  if (lastDot === -1) return null
  return path.slice(lastDot + 1).toLowerCase()
}

export function formatPageRange(result: SearchResult): string | null {
  const { page_start, page_end } = result
  if (page_start == null && page_end == null) return null
  if (page_start != null && page_end != null) {
    if (page_start === page_end) return `Page ${page_start}`
    return `Pages ${page_start}–${page_end}`
  }
  if (page_start != null) return `From page ${page_start}`
  if (page_end != null) return `Up to page ${page_end}`
  return null
}

export type AggregatedResult = SearchResult & {
  hit_pages?: string[]
}

export function dedupeResultsByFile(
  results: SearchResult[],
  k: number,
): AggregatedResult[] {
  const byFile = new Map<
    string,
    {
      result: SearchResult
      pages: Set<string>
    }
  >()

  for (const result of results) {
    const key = result.filepath ?? result.id
    const pageLabel = formatPageRange(result)
    const existing = byFile.get(key)

    if (!existing) {
      const pages = new Set<string>()
      if (pageLabel) pages.add(pageLabel)
      byFile.set(key, { result, pages })
    } else {
      if (result.score > existing.result.score) {
        existing.result = result
      }
      if (pageLabel) existing.pages.add(pageLabel)
    }
  }

  const aggregated: AggregatedResult[] = []
  for (const { result, pages } of byFile.values()) {
    aggregated.push({ ...result, hit_pages: Array.from(pages) })
  }

  aggregated.sort((a, b) => b.score - a.score)
  return aggregated.slice(0, k)
}
