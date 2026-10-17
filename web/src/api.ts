// Client for the Go API. In dev, Vite proxies /api/* to localhost:8080.
const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

export type Source = {
  document_id: string
  filename: string
  kind: string
  section_title?: string
  page: number
  chunk_index: number
  score: number
  text: string
}

export type DocumentInfo = {
  document_id: string
  filename: string
  kind: string
  pages: number
  chunks: number
  created_at: string
}

// unitLabel names what a page number counts for a file kind.
export function unitLabel(kind: string): string {
  if (kind === 'pptx') return 'slide'
  if (kind === 'image') return 'gambar'
  if (kind === 'pdf' || !kind) return 'hlm'
  return 'bagian'
}

export class ApiError extends Error {
  readonly status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function errorFrom(res: Response): Promise<ApiError> {
  try {
    const body = await res.json()
    if (body?.error) return new ApiError(body.error, res.status)
  } catch {
    // not JSON
  }
  return new ApiError(`HTTP ${res.status}`, res.status)
}

// The admin token lives in this browser only; it is a convenience, the
// server is what enforces it.
const TOKEN_KEY = 'rag-chatbot:admin-token'

export function getAdminToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    return ''
  }
}

export function setAdminToken(token: string) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    // storage unavailable; the token is simply not remembered
  }
}

function adminHeaders(): HeadersInit {
  const token = getAdminToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export type AppConfig = { admin_required: boolean; extensions: string[]; max_upload_mb: number }

export async function getConfig(): Promise<AppConfig> {
  const res = await fetch(`${API_BASE}/config`)
  if (!res.ok) throw await errorFrom(res)
  return res.json()
}

export async function listDocuments(signal?: AbortSignal): Promise<DocumentInfo[]> {
  const res = await fetch(`${API_BASE}/documents`, { signal })
  if (!res.ok) throw await errorFrom(res)
  return res.json()
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    headers: adminHeaders(),
  })
  // 404 means it is already gone, which is what the caller wanted.
  if (!res.ok && res.status !== 404) throw await errorFrom(res)
}

/** Thrown when the same file was already uploaded; carries the existing document. */
export class DuplicateError extends ApiError {
  readonly documentId: string
  constructor(message: string, documentId: string) {
    super(message, 409)
    this.documentId = documentId
  }
}

export type IngestAccepted = { job_id: string; document_id: string; filename: string }

export async function ingestFile(file: File, signal?: AbortSignal): Promise<IngestAccepted> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/ingest`, { method: 'POST', body: form, headers: adminHeaders(), signal })
  if (res.status === 409) {
    const body = await res.json()
    throw new DuplicateError(body.error, body.document_id)
  }
  if (!res.ok) throw await errorFrom(res)
  return res.json()
}

export type JobStatus = {
  job_id: string
  document_id: string
  filename: string
  status: 'queued' | 'extracting' | 'embedding' | 'done' | 'error'
  done: number
  total: number
  error?: string
  document?: DocumentInfo
}

export async function getJob(id: string, signal?: AbortSignal): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/ingest/${encodeURIComponent(id)}`, { signal })
  if (!res.ok) throw await errorFrom(res)
  return res.json()
}

type ChatHandlers = {
  onToken: (text: string) => void
  onSources: (sources: Source[]) => void
}

// streamChat posts a question and dispatches Server-Sent Events as they arrive.
// It resolves on 'done' and rejects on 'error', HTTP errors or abort.
export type Turn = { role: 'user' | 'assistant'; content: string }

export async function streamChat(
  req: { question: string; document_id?: string; history?: Turn[] },
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
