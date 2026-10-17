import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  ApiError,
  deleteDocument,
  getAdminToken,
  DuplicateError,
  getConfig,
  getJob,
  ingestFile,
  listDocuments,
  setAdminToken,
  sourceUrl,
  streamChat,
  unitLabel,
  type AppConfig,
  type JobStatus,
  type DocumentInfo,
  type Source,
  type Turn,
} from './api'
import './App.css'

type Message = {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  error?: string
  streaming?: boolean
}

let nextId = 1

const errorText = (err: unknown) => (err instanceof Error ? err.message : String(err))

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [token, setToken] = useState(getAdminToken)
  const [tokenDraft, setTokenDraft] = useState('')
  // Upload and delete are shown when the server needs no token or one is saved.
  // A wrong token is caught by the server's 401 below.
  const canEdit = config !== null && (!config.admin_required || token !== '')

  function saveToken(value: string) {
    setAdminToken(value)
    setToken(value)
  }

  // A 401 means the saved token is wrong or was rotated: forget it.
  function handleAuthError(err: unknown): string {
    if (err instanceof ApiError && err.status === 401) {
      saveToken('')
      return 'Token admin salah. Masukkan lagi.'
    }
    return errorText(err)
  }

  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [docsLoading, setDocsLoading] = useState(true)
  const [docsError, setDocsError] = useState('')
  const [deleting, setDeleting] = useState<string | null>(null)
  const [activeDoc, setActiveDoc] = useState<string>('') // '' = all documents
  const [uploading, setUploading] = useState(false)
  const [job, setJob] = useState<JobStatus | null>(null)
  const [uploadError, setUploadError] = useState('')
  const [dragOver, setDragOver] = useState(false)

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // docsLoading starts true; the refresh button sets it before calling this.
  async function refreshDocs(signal?: AbortSignal) {
    try {
      const list = await listDocuments(signal)
      setDocs(list)
      setDocsError('')
      // Drop a selection whose document no longer exists.
      setActiveDoc((cur) => (cur && !list.some((d) => d.document_id === cur) ? '' : cur))
    } catch (err) {
      if (!signal?.aborted) setDocsError(errorText(err))
    } finally {
      if (!signal?.aborted) setDocsLoading(false)
    }
  }

  useEffect(() => {
    const ctrl = new AbortController()
    // State is only set after the fetch resolves, which is what effects are for.
    // eslint-disable-next-line react/set-state-in-effect
    refreshDocs(ctrl.signal)
    getConfig()
      .then(setConfig)
      .catch(() => setConfig({ admin_required: false, extensions: ['.pdf'], max_upload_mb: 20 }))
    return () => ctrl.abort()
  }, [])
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  async function upload(file: File) {
    const exts = config?.extensions ?? ['.pdf']
    const name = file.name.toLowerCase()
    if (!exts.some((e) => name.endsWith(e))) {
      setUploadError(`Format belum didukung. Pilih: ${exts.join(', ')}`)
      return
    }
    const maxMB = config?.max_upload_mb ?? 20
    if (file.size > maxMB << 20) {
      setUploadError(`File lebih dari ${maxMB} MB.`)
      return
    }
    setUploading(true)
    setUploadError('')
    setJob(null)
    try {
      const accepted = await ingestFile(file)
      // Indexing runs on the server; poll until it finishes.
      let status: JobStatus
      for (;;) {
        status = await getJob(accepted.job_id)
        setJob(status)
        if (status.status === 'done' || status.status === 'error') break
        await new Promise((r) => setTimeout(r, 800))
      }
      if (status.status === 'error') {
        setUploadError(status.error ?? 'Pemrosesan gagal.')
        return
      }
      await refreshDocs()
      setActiveDoc(accepted.document_id)
    } catch (err) {
      if (err instanceof DuplicateError) {
        setUploadError(`${err.message}. Dokumen itu dipilih.`)
        setActiveDoc(err.documentId)
        return
      }
      setUploadError(handleAuthError(err))
    } finally {
      setUploading(false)
      setJob(null)
    }
  }

  async function removeDoc(doc: DocumentInfo) {
    if (!window.confirm(`Hapus "${doc.filename}" beserta semua chunk-nya dari Qdrant?`)) return
    setDeleting(doc.document_id)
    try {
      await deleteDocument(doc.document_id)
      setDocs((d) => d.filter((x) => x.document_id !== doc.document_id))
      if (activeDoc === doc.document_id) setActiveDoc('')
      setDocsError('')
    } catch (err) {
      setDocsError(`Gagal menghapus: ${handleAuthError(err)}`)
    } finally {
      setDeleting(null)
    }
  }

  function patchMessage(id: number, patch: (m: Message) => Partial<Message>) {
    setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...patch(m) } : m)))
  }

  async function ask(e?: FormEvent) {
    e?.preventDefault()
    const question = input.trim()
    if (!question || busy) return

    const history = buildHistory(messages)
    const userId = nextId++
    const assistantId = nextId++
    setMessages((ms) => [
      ...ms,
      { id: userId, role: 'user', content: question },
      { id: assistantId, role: 'assistant', content: '', streaming: true },
    ])
    setInput('')
    setBusy(true)

    const ctrl = new AbortController()
    abortRef.current = ctrl
    try {
      await streamChat(
        { question, document_id: activeDoc || undefined, history },
        {
          onToken: (t) => patchMessage(assistantId, (m) => ({ content: m.content + t })),
          onSources: (sources) => patchMessage(assistantId, () => ({ sources })),
        },
        ctrl.signal,
      )
    } catch (err) {
      const aborted = ctrl.signal.aborted
      patchMessage(assistantId, () => ({
        error: aborted ? 'Dihentikan.' : errorText(err),
      }))
    } finally {
      patchMessage(assistantId, () => ({ streaming: false }))
      abortRef.current = null
      setBusy(false)
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      ask()
    }
  }

  const activeName = docs.find((d) => d.document_id === activeDoc)?.filename

  return (
    <div className="app">
      <aside className="sidebar">
        <h1 className="brand">RAG Chatbot</h1>

        {config?.admin_required && !token && (
          <form
            className="admin-login"
            onSubmit={(e) => {
              e.preventDefault()
              saveToken(tokenDraft.trim())
              setTokenDraft('')
              setUploadError('')
            }}
          >
            <label htmlFor="admin-token">Token admin untuk upload & hapus</label>
            <div className="admin-row">
              <input
                id="admin-token"
                type="password"
                value={tokenDraft}
                onChange={(e) => setTokenDraft(e.target.value)}
                autoComplete="current-password"
              />
              <button type="submit" disabled={!tokenDraft.trim()}>
                Masuk
              </button>
            </div>
          </form>
        )}
        {config?.admin_required && token && (
          <button className="admin-logout" onClick={() => saveToken('')}>
            Keluar dari mode admin
          </button>
        )}

        {canEdit && (
        <div
          className={`dropzone${dragOver ? ' over' : ''}${uploading ? ' busy' : ''}`}
          onClick={() => !uploading && fileRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            const file = e.dataTransfer.files[0]
            if (file && !uploading) upload(file)
          }}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && fileRef.current?.click()}
        >
          {uploading ? (
            <UploadProgress job={job} />
          ) : (
            <>
              <strong>Upload dokumen</strong>
              <span>
                {(config?.extensions ?? ['.pdf']).filter((e) => e !== '.markdown').map((e) => e.slice(1).toUpperCase()).join(', ')} · maks{' '}
                {config?.max_upload_mb ?? 20} MB
              </span>
            </>
          )}
          <input
            ref={fileRef}
            type="file"
            accept={(config?.extensions ?? ['.pdf']).join(',')}
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) upload(file)
              e.target.value = ''
            }}
          />
        </div>
        )}
        {uploadError && <p className="upload-error">{uploadError}</p>}

        <h2 className="section-title">
          Dokumen
          <button className="refresh" onClick={() => {
              setDocsLoading(true)
              refreshDocs()
            }} disabled={docsLoading} aria-label="Muat ulang daftar dokumen">
            {docsLoading ? '…' : '↻'}
          </button>
        </h2>
        {docsError && <p className="upload-error">{docsError}</p>}
        {!docsLoading && !docsError && docs.length === 0 && <p className="doc-empty">Belum ada dokumen.</p>}
        <ul className="doc-list">
          <li>
            <button className={activeDoc === '' ? 'doc active' : 'doc'} onClick={() => setActiveDoc('')}>
              <span className="doc-name">Semua dokumen</span>
            </button>
          </li>
          {docs.map((d) => (
            <li key={d.document_id}>
              <button
                className={activeDoc === d.document_id ? 'doc active' : 'doc'}
                onClick={() => setActiveDoc(d.document_id)}
                title={d.filename}
              >
                <span className="doc-name">{d.filename}</span>
                <span className="doc-meta">
                  {d.pages} {unitLabel(d.kind)} · {d.chunks} chunk
                </span>
              </button>
              {canEdit && (
                <button
                  className="forget"
                  onClick={() => removeDoc(d)}
                  disabled={deleting === d.document_id}
                  aria-label={`Hapus ${d.filename}`}
                  title="Hapus dokumen"
                >
                  {deleting === d.document_id ? '…' : '×'}
                </button>
              )}
            </li>
          ))}
        </ul>
      </aside>

      <main className="chat">
        <header className="chat-header">
          <span>
            Bertanya ke: <strong>{activeName ?? 'semua dokumen'}</strong>
          </span>
          <button className="new-chat" onClick={() => setMessages([])} disabled={busy || messages.length === 0}>
            Chat baru
          </button>
        </header>

        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <p>Upload PDF di samping, lalu tanyakan isinya.</p>
              <p className="hint">Jawaban menyertakan sitasi seperti [1] yang merujuk ke halaman sumber.</p>
            </div>
          )}
          {messages.map((m) => (
            <MessageView key={m.id} message={m} />
          ))}
          <div ref={bottomRef} />
        </div>

        <form className="composer" onSubmit={ask}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Tulis pertanyaan… (Enter kirim, Shift+Enter baris baru)"
            rows={2}
            maxLength={2000}
          />
          {busy ? (
            <button type="button" className="secondary" onClick={() => abortRef.current?.abort()}>
              Stop
            </button>
          ) : (
            <button type="submit" disabled={!input.trim()}>
              Kirim
            </button>
          )}
        </form>
      </main>
    </div>
  )
}

function MessageView({ message: m }: { message: Message }) {
  const [openSource, setOpenSource] = useState<number | null>(null)

  return (
    <div className={`msg ${m.role}`}>
      <div className={`bubble${m.streaming && m.content ? ' streaming' : ''}`}>
        {m.role === 'assistant' ? <AnswerMarkdown text={m.content} sources={m.sources} open={setOpenSource} /> : m.content}
        {m.streaming && !m.content && <span className="typing">Mencari jawaban…</span>}
        {m.error && <p className="msg-error">{m.error}</p>}
      </div>

      {m.sources && m.sources.length > 0 && (
        <ol className="sources">
          {/* Only cited sources arrive; each keeps the number the answer uses. */}
          {m.sources.map((s) => (
            <li key={s.n} className={openSource === s.n ? 'open' : ''}>
              <div className="source-row">
                <button onClick={() => setOpenSource(openSource === s.n ? null : s.n)}>
                  <span className="cite">{s.n}</span>
                  {s.filename}, {unitLabel(s.kind)} {s.page}
                  {s.section_title && <span className="section-title-tag">{s.section_title}</span>}
                  <span className="score">{s.score > 0 ? `${(s.score * 100).toFixed(0)}%` : 'awal dokumen'}</span>
                </button>
                <a className="open-file" href={sourceUrl(s)} target="_blank" rel="noopener noreferrer" title="Buka file asli">
                  Buka ↗
                </a>
              </div>
              {openSource === s.n && <blockquote>{s.text}</blockquote>}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

// The server keeps no state, so each request carries the last few exchanges.
// Only complete question/answer pairs are sent; failed or stopped answers are skipped.
const MAX_HISTORY = 6

function buildHistory(messages: Message[]): Turn[] {
  const turns: Turn[] = []
  for (let i = 0; i + 1 < messages.length; i++) {
    const q = messages[i]
    const a = messages[i + 1]
    // Answers without sources are "not found" replies; they add nothing but
    // would steer the next retrieval toward an unanswerable question.
    if (q.role === 'user' && a.role === 'assistant' && !a.error && !a.streaming && a.content && a.sources?.length) {
      turns.push({ role: 'user', content: q.content }, { role: 'assistant', content: a.content })
      i++
    }
  }
  return turns.slice(-MAX_HISTORY)
}

// Citations "[2]" (or fullwidth "【2】", which gpt-oss uses) become links to
// "#cite-2" before Markdown parsing, so the link renderer below can turn them
// into buttons. "[x](url)" links and "[x]: url" definitions are left alone.
const CITATION = /\[(\d+)\](?![(:[])|【(\d+)】/g

function linkCitations(text: string): string {
  return text.replace(CITATION, (_, a, b) => `[${a ?? b}](#cite-${a ?? b})`)
}

function AnswerMarkdown({ text, sources, open }: { text: string; sources?: Source[]; open: (i: number) => void }) {
  return (
    <div className="md">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            const n = href?.startsWith('#cite-') ? Number(href.slice(6)) : 0
            if (n) {
              const s = sources?.find((x) => x.n === n)
              // Sources arrive after the answer has streamed; until then the
              // marker is shown but not clickable.
              return s ? (
                <button className="cite" onClick={() => open(n)} title={`${s.filename}, ${unitLabel(s.kind)} ${s.page}`}>
                  {n}
                </button>
              ) : (
                <span className="cite">{n}</span>
              )
            }
            return (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            )
          },
        }}
      >
        {linkCitations(text)}
      </Markdown>
    </div>
  )
}

const STAGES: Record<JobStatus['status'], string> = {
  queued: 'Menunggu antrean…',
  extracting: 'Membaca isi file…',
  embedding: 'Membuat embedding…',
  done: 'Selesai',
  error: 'Gagal',
}

function UploadProgress({ job }: { job: JobStatus | null }) {
  const pct = job && job.total > 0 ? Math.round((job.done / job.total) * 100) : 0
  return (
    <div className="progress" role="status" aria-live="polite">
      <span>{job ? STAGES[job.status] : 'Mengunggah…'}</span>
      <div className="progress-bar">
        <div style={{ width: `${job?.status === 'embedding' ? pct : job ? 5 : 2}%` }} />
      </div>
      {job?.status === 'embedding' && (
        <span className="progress-count">
          {job.done}/{job.total} chunk
        </span>
      )}
    </div>
  )
}
