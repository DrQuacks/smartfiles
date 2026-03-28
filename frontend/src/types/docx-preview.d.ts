declare module 'docx-preview' {
  export function renderAsync(
    arrayBuffer: ArrayBuffer,
    container: HTMLElement,
    styleOptions?: unknown,
    options?: unknown,
  ): Promise<void>
}
