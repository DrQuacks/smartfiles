import { useEffect, useRef, useState } from 'react'

export type DocxPreviewProps = {
  fileUrl: string
}

export default function DocxPreview({ fileUrl }: DocxPreviewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const container = containerRef.current
    if (!container) return

    container.innerHTML = ''
    setError(null)

    const load = async () => {
      try {
        const res = await fetch(fileUrl)
        if (!res.ok) {
          throw new Error(`Failed to fetch document: ${res.status}`)
        }
        const blob = await res.blob()
        const arrayBuffer = await blob.arrayBuffer()

        // Import here to avoid bundling docx-preview on server side.
        const { renderAsync } = await import('docx-preview')
        if (cancelled) return
        await renderAsync(arrayBuffer, container)
      } catch (err) {
        if (cancelled) return
        console.error(err)
        setError('Unable to render Word document inline.')
      }
    }

    load()

    return () => {
      cancelled = true
      if (container) {
        container.innerHTML = ''
      }
    }
  }, [fileUrl])

  if (error) {
    return <p className="preview-fallback-text">{error}</p>
  }

  return <div ref={containerRef} className="docx-preview-container" />
}
