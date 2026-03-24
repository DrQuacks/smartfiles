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
