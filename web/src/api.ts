// Client for the Go API. In dev, Vite proxies /api/* to localhost:8080.
const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

export type Source = {
  document_id: string
  filename: string
  page: number
  chunk_index: number
  score: number
  text: string
}

export type IngestResult = {
  document_id: string
  filename: string
  pages: number
  chunks: number
}

async function errorFrom(res: Response): Promise<Error> {
  try {
    const body = await res.json()
    if (body?.error) return new Error(body.error)
  } catch {
    // not JSON
  }
  return new Error(`HTTP ${res.status}`)
}

export type DocumentInfo = IngestResult & { created_at: string }

export async function listDocuments(signal?: AbortSignal): Promise<DocumentInfo[]> {
  const res = await fetch(`${API_BASE}/documents`, { signal })
  if (!res.ok) throw await errorFrom(res)
  return res.json()
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(id)}`, { method: 'DELETE' })
  // 404 means it is already gone, which is what the caller wanted.
  if (!res.ok && res.status !== 404) throw await errorFrom(res)
}

export async function ingestPDF(file: File, signal?: AbortSignal): Promise<IngestResult> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/ingest`, { method: 'POST', body: form, signal })
  if (!res.ok) throw await errorFrom(res)
  return res.json()
}

type ChatHandlers = {
  onToken: (text: string) => void
  onSources: (sources: Source[]) => void
}

// streamChat posts a question and dispatches Server-Sent Events as they arrive.
// It resolves on "done" and rejects on "error", HTTP errors or abort.
export async function streamChat(
  req: { question: string; document_id?: string },
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(req),
    signal,
  })
  if (!res.ok) throw await errorFrom(res)
  if (!res.body) throw new Error('Respons tidak bisa di-stream')

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value

    // Events are separated by a blank line.
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const raw = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const { event, data } = parseEvent(raw)
      if (!data) continue

      const payload = JSON.parse(data)
      switch (event) {
        case 'token':
          handlers.onToken(payload.text)
          break
        case 'sources':
          handlers.onSources(payload)
          break
        case 'error':
          throw new Error(payload.error ?? 'Terjadi kesalahan')
        case 'done':
          return
      }
    }
  }
  throw new Error('Koneksi terputus sebelum jawaban selesai')
}

function parseEvent(raw: string): { event: string; data: string } {
  let event = 'message'
  const data: string[] = []
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
  }
  return { event, data: data.join('\n') }
}
