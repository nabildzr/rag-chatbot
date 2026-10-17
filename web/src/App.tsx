import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from 'react'
import {
  deleteDocument,
  ingestPDF,
  listDocuments,
  streamChat,
  type DocumentInfo,
  type Source,
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
  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [docsLoading, setDocsLoading] = useState(true)
  const [docsError, setDocsError] = useState('')
  const [deleting, setDeleting] = useState<string | null>(null)
  const [activeDoc, setActiveDoc] = useState<string>('') // '' = all documents
  const [uploading, setUploading] = useState(false)
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
    return () => ctrl.abort()
  }, [])
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  async function upload(file: File) {
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('File harus PDF.')
      return
    }
    setUploading(true)
    setUploadError('')
    try {
      const doc = await ingestPDF(file)
      await refreshDocs()
      setActiveDoc(doc.document_id)
    } catch (err) {
      setUploadError(errorText(err))
    } finally {
      setUploading(false)
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
      setDocsError(`Gagal menghapus: ${errorText(err)}`)
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
        { question, document_id: activeDoc || undefined },
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
            <span>Memproses PDF…</span>
          ) : (
            <>
              <strong>Upload PDF</strong>
              <span>Klik atau drag file ke sini (maks 20 MB)</span>
            </>
          )}
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf,.pdf"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) upload(file)
              e.target.value = ''
            }}
          />
        </div>
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
                  {d.pages} hlm · {d.chunks} chunk
                </span>
              </button>
              <button
                className="forget"
                onClick={() => removeDoc(d)}
                disabled={deleting === d.document_id}
                aria-label={`Hapus ${d.filename}`}
                title="Hapus dokumen"
              >
                {deleting === d.document_id ? '…' : '×'}
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className="chat">
        <header className="chat-header">
          Bertanya ke: <strong>{activeName ?? 'semua dokumen'}</strong>
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
      <div className="bubble">
        {m.role === 'assistant' ? withCitations(m.content, m.sources, setOpenSource) : m.content}
        {m.streaming && !m.content && <span className="typing">Mencari jawaban…</span>}
        {m.streaming && m.content && <span className="cursor" />}
        {m.error && <p className="msg-error">{m.error}</p>}
      </div>

      {m.sources && m.sources.length > 0 && (
        <ol className="sources">
          {m.sources.map((s, i) => (
            <li key={i} className={openSource === i ? 'open' : ''}>
              <button onClick={() => setOpenSource(openSource === i ? null : i)}>
                <span className="cite">{i + 1}</span>
                {s.filename}, hlm {s.page}
                <span className="score">{(s.score * 100).toFixed(0)}%</span>
              </button>
              {openSource === i && <blockquote>{s.text}</blockquote>}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

// withCitations turns "[2]" in the answer into a button that opens source 2.
// Some models (gpt-oss) write fullwidth brackets "【2】", so accept both.
function withCitations(text: string, sources: Source[] | undefined, open: (i: number) => void): ReactNode[] {
  return text.split(/(\[\d+\]|【\d+】)/g).map((part, i) => {
    const digits = /^(?:\[(\d+)\]|【(\d+)】)$/.exec(part)?.slice(1).find(Boolean)
    const n = digits ? Number(digits) : 0
    if (sources && n >= 1 && n <= sources.length) {
      return (
        <button key={i} className="cite" onClick={() => open(n - 1)} title={`${sources[n - 1].filename}, hlm ${sources[n - 1].page}`}>
          {n}
        </button>
      )
    }
    return part
  })
}
