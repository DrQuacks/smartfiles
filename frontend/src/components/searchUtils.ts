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
  page_summary?: string
}

function collectPagesFromResult(set: Set<number>, result: SearchResult): void {
  const { page_start, page_end } = result
  if (page_start == null && page_end == null) return
  if (page_start != null && page_end != null) {
    const start = Math.min(page_start, page_end)
    const end = Math.max(page_start, page_end)
    for (let p = start; p <= end; p += 1) {
      set.add(p)
    }
    return
  }
  if (page_start != null) {
    set.add(page_start)
  } else if (page_end != null) {
    set.add(page_end)
  }
}

export function formatPageSummaryFromPages(pages: number[]): string | null {
  if (!pages.length) return null

  const uniqueSorted = Array.from(new Set(pages)).sort((a, b) => a - b)
  if (!uniqueSorted.length) return null

  const ranges: { start: number; end: number }[] = []
  let start = uniqueSorted[0]
  let prev = uniqueSorted[0]

  for (let i = 1; i < uniqueSorted.length; i += 1) {
    const p = uniqueSorted[i]
    if (p === prev + 1) {
      prev = p
      continue
    }
    ranges.push({ start, end: prev })
    start = p
    prev = p
  }
  ranges.push({ start, end: prev })

  if (ranges.length === 1) {
    const r = ranges[0]
    if (r.start === r.end) return `Page ${r.start}`
    return `Pages ${r.start}–${r.end}`
  }

  const parts = ranges.map((r) =>
    r.start === r.end ? `${r.start}` : `${r.start}–${r.end}`,
  )
  return `Pages ${parts.join(', ')}`
}

export function dedupeResultsByFile(
  results: SearchResult[],
  k: number,
): AggregatedResult[] {
  const byFile = new Map<
    string,
    {
      result: SearchResult
      pages: Set<number>
    }
  >()

  for (const result of results) {
    const key = result.filepath ?? result.id
    const existing = byFile.get(key)

    if (!existing) {
      const pages = new Set<number>()
      collectPagesFromResult(pages, result)
      byFile.set(key, { result, pages })
    } else {
      if (result.score > existing.result.score) {
        existing.result = result
      }
      collectPagesFromResult(existing.pages, result)
    }
  }

  const aggregated: AggregatedResult[] = []
  for (const { result, pages } of byFile.values()) {
    const page_summary = formatPageSummaryFromPages(Array.from(pages))
    aggregated.push({ ...result, page_summary })
  }

  aggregated.sort((a, b) => b.score - a.score)
  return aggregated.slice(0, k)
}
