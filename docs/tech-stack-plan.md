# Tech Stack Plan — RAG Chatbot (Hybrid, Portfolio Project)

> Chatbot tanya-jawab berbasis dokumen PDF. Arsitektur hybrid: vector DB +
> embedding jalan self-hosted di VPS, LLM pakai API (swappable).

## Arsitektur

```
[ Vite + React SPA ] ──POST /chat (SSE stream)──> [ Go API ]
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        │                               │                               │
                 [ Qdrant ]                    [ Ollama ]                      [ Groq API ]
              (Docker, VPS)              (VPS, nomic-embed-text)              (LLM cloud)
              vector DB                   embedding lokal                      generasi jawaban
```

**Alur ingest:** Upload PDF → ekstrak teks → chunking (±500 token, overlap 50)
→ embedding via Ollama → simpan vector + metadata di Qdrant.

**Alur chat:** Pertanyaan → embedding via Ollama → similarity search top-k (5)
di Qdrant → chunk + pertanyaan dirangkai jadi prompt → Groq API (streaming)
→ jawaban + sitasi chunk ke frontend.

## Komponen

| Lapisan     | Teknologi | Keterangan |
|-------------|-----------|------------|
| Frontend    | Vite + React + TypeScript (SPA) | Chat UI, streaming via `fetch` reader; dev proxy `/api` ke Go |
| Backend     | Go (stdlib `net/http` + chi/gorilla/mux) | Single binary, deploy mudah |
| Vector DB   | Qdrant (Docker, di VPS) | Satu collection `chunks`, filter per `document_id`; payload = document_id, filename, page, chunk_index, text, created_at |
| Embedding   | Ollama `nomic-embed-text` (di VPS) | ~274MB, jalan mulus di CPU, gratis |
| LLM         | Groq API (model `openai/gpt-oss-120b`) | Free tier cukup untuk demo; gampang diganti ke Ollama lokal / OpenAI |
| PDF parse   | Go: `ledongthuc/pdf` atau `unidoc/unipdf` | Ekstrak teks per halaman |

## Setup VPS

`docker-compose.yml` (di VPS):

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"      # REST API
      - "6334:6334"      # gRPC
    volumes:
      - qdrant_data:/qdrant/storage

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  qdrant_data:
  ollama_data:
```

Setelah compose up:
```bash
docker exec <ollama-container> ollama pull bge-m3
```

> Catatan: LLM 7B+ sengaja TIDAK dijalankan di VPS (CPU 4 core terlalu lambat
> untuk generasi). VPS hanya untuk Qdrant + embedding yang ringan.

## Environment Variables (JANGAN hardcode / commit!)

Buat file `.env` di root project backend (dan pastikan masuk `.gitignore`):

```bash
# .env — JANGAN pernah commit file ini ke Git!
GROQ_API_KEY=isi_dengan_api_key_kamu_di_sini
GROQ_MODEL=openai/gpt-oss-120b
OLLAMA_URL=http://localhost:11434
QDRANT_URL=http://localhost:6333
PORT=8080
```

Go membaca via `os.Getenv("GROQ_API_KEY")`. Di production (VPS), set sebagai
environment variable sistem / di systemd unit / docker env — bukan di file.

## Struktur Project (Go backend)

```
rag-chatbot/
├── docker-compose.yml
├── .env.example          # template TANPA nilai rahasia (boleh commit)
├── .gitignore            # wajib ada: .env
├── go.mod
├── cmd/api/main.go       # entrypoint
├── internal/
│   ├── handler/          # POST /ingest, POST /chat, GET /documents
│   ├── rag/              # chunking, prompt builder
│   ├── vector/           # client Qdrant (REST via net/http)
│   ├── embed/            # client Ollama embeddings
│   └── llm/              # client Groq (OpenAI-compatible /chat/completions, stream)
└── web/                  # Vite + React frontend
    ├── vite.config.ts    # proxy /api -> localhost:8080
    └── src/App.tsx       # chat UI
```

## Endpoint API

- `POST /ingest` — multipart PDF → `{ "document_id": "...", "chunks": 42 }`
- `POST /chat` — `{ "question": "...", "document_id": "..." }` → SSE stream
  token + di akhir kirim `sources: [...]` (chunk yang dipakai sebagai sitasi)
- `GET /documents` — daftar dokumen yang sudah di-ingest
- `DELETE /documents/{id}` — hapus dokumen + vector-nya

## Langkah Pengerjaan

1. `docker compose up -d` di VPS → verifikasi Qdrant dashboard di `:6333/dashboard`
2. Pull `nomic-embed-text` di Ollama, tes embedding 1 kalimat
3. Go: client Qdrant (create collection, upsert, search) — tes manual
4. Go: `/ingest` — PDF → chunk → embed → Qdrant
5. Go: `/chat` — retrieve → prompt → Groq streaming (SSE)
6. Vite + React: chat UI + upload PDF
7. README + screenshot/demo GIF → push ke GitHub → pin repo

## Keamanan API Key

- API key HANYA lewat environment variable, tidak pernah di-hardcode,
  tidak pernah di-commit, tidak pernah dishare di chat/screenshot.
- File `.env.example` hanya berisi nama variable kosong sebagai dokumentasi.
- Kalau sebuah key pernah bocor (terpaste di chat, ter-commit, dsb):
  revoke di dashboard provider (console.groq.com → API Keys → revoke)
  lalu generate key baru dan update `.env`.
