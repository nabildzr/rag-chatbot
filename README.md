# RAG Chatbot

[![CI](https://github.com/nabildzr/rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/nabildzr/rag-chatbot/actions/workflows/ci.yml)

Chatbot tanya-jawab berbasis dokumen. Upload PDF, Word, PowerPoint, Markdown, teks,
atau gambar/scan, lalu tanyakan isinya, juga dalam bahasa yang berbeda dari
dokumennya. Jawaban di-stream token demi token dan menyertakan sitasi `[n]` yang
membuka halaman, slide, atau bagian sumbernya.

Vector DB dan embedding jalan self-hosted (Docker). LLM memakai API cloud yang
OpenAI-compatible (default: Groq), jadi mudah diganti.

![Tanya-jawab atas PDF dengan sitasi halaman sumber](docs/screenshot.png)

## Fitur

- **Format:** PDF, DOCX, PPTX, TXT, Markdown. Ada juga PNG/JPG dan PDF hasil scan lewat
  OCR (Tesseract, bahasa Indonesia + Inggris).
- **Pencarian hybrid:** embedding multibahasa `bge-m3` + BM25, digabung dengan
  reciprocal rank fusion. Istilah persis (kode, angka, nama) tetap ketemu.
- **Lintas bahasa:** tanya dalam bahasa Indonesia tentang dokumen berbahasa Inggris,
  atau sebaliknya.
- **Percakapan multi-turn:** pertanyaan lanjutan ("kalau yang lembur?") memakai konteks
  sebelumnya.
- **Sitasi yang jujur:** hanya sumber yang benar-benar dikutip yang ditampilkan.
  Jawaban "tidak ditemukan" tidak membawa sumber. Klik "Buka" untuk melihat file asli
  (PDF langsung ke halamannya).
- **Ingest di background** dengan progress bar, deteksi file duplikat (SHA-256), dan
  file asli disimpan.
- **Siap publik:** token admin untuk upload/hapus, rate limit per IP, image Docker
  non-root, Qdrant/Ollama tidak terbuka ke internet.
- **Terukur:** set evaluasi 47 pertanyaan (`cmd/eval`), dipakai untuk memilih model
  dan threshold. Hasilnya ada di [eval/RESULTS.md](eval/RESULTS.md).

## Hasil evaluasi

Retrieval diukur dengan 47 pertanyaan yang jawabannya diketahui, dicari di semua
dokumen sekaligus (detail di [eval/RESULTS.md](eval/RESULTS.md)):

| konfigurasi | bahasa sama | lintas bahasa | kata kunci persis | total hit@1 |
|---|---|---|---|---|
| nomic-embed-text, dense | 76% | 20% | – | 50% |
| bge-m3, dense | 100% | 93% | 80% | 93% |
| **bge-m3 + BM25 hybrid (default)** | **100%** | **93%** | **100%** | **98%** |

Reranker cross-encoder juga diuji. Di CPU hasilnya lebih buruk (model kecil) atau
terlalu lambat (±60 detik/pertanyaan, model besar), jadi fitur ini opsional untuk host
dengan GPU.

## Arsitektur

```
[ Vite + React SPA ] ──/api──> [ Go API (net/http) ] ──> [ Groq API ] LLM, streaming
                                  │        │        │
                             [ Qdrant ] [ Ollama ] [ Tesseract ]      [ TEI reranker ]
                         dense + BM25    bge-m3     OCR, opsional     opsional, GPU
```

**Ingest** (`POST /ingest`, async): deteksi format dari isi file → simpan file asli →
balas `202` dengan job ID. Di background: ekstrak teks per halaman/slide/bagian (OCR
untuk gambar dan halaman scan) → chunk ±375 kata (overlap 40) → embedding dense
(Ollama) + vektor BM25 → upsert ke Qdrant dengan `ready=false` → tandai `ready=true`
setelah semua masuk. Kalau gagal, chunk dan file dihapus lagi, jadi tidak ada
dokumen setengah jadi.

**Chat** (`POST /chat`): cari top-20 dense dan top-20 BM25 (untuk pertanyaan
lanjutan juga dengan pertanyaan sebelumnya) → gabungkan dengan RRF → saring dengan
skor cosine (`MIN_SCORE`, `SCORE_MARGIN`; chunk yang memuat semua kata kunci
selalu lolos) → prompt dengan konteks bernomor → Groq streaming lewat SSE → setelah
selesai, kirim hanya sumber yang dikutip. Kalau tidak ada chunk yang relevan, API
langsung menjawab "tidak ditemukan" tanpa memanggil LLM. Pertanyaan ringkasan
("apa inti dokumen ini?") pada satu dokumen memakai bagian awal dokumen.

## Stack

| Lapisan   | Teknologi |
|-----------|-----------|
| Frontend  | Vite + React + TypeScript, react-markdown |
| Backend   | Go, stdlib `net/http` (tanpa framework) |
| Vector DB | Qdrant: dense + sparse vector, IDF dihitung Qdrant |
| Embedding | Ollama `bge-m3` (1024 dimensi, multibahasa) |
| LLM       | Groq `openai/gpt-oss-120b`, endpoint OpenAI-compatible |
| Ekstraksi | `ledongthuc/pdf` (dengan rekonstruksi spasi), parser DOCX/PPTX sendiri, Tesseract + Poppler |
| Reranker  | Hugging Face Text Embeddings Inference, opsional |

## Menjalankan secara lokal

Prasyarat: Go 1.27+, Node.js 20+, Docker. Opsional: Tesseract dan Poppler di PATH
untuk OCR.

1. Jalankan Qdrant dan Ollama, lalu pull model embedding:

   ```bash
   docker compose up -d
   docker compose exec ollama ollama pull bge-m3
   ```

   Dashboard Qdrant tersedia di http://localhost:6333/dashboard.

2. Siapkan environment variable, lalu isi `GROQ_API_KEY` (buat di
   https://console.groq.com). File `.env` sudah di-`.gitignore`.

   ```bash
   cp .env.example .env
   ```

3. Jalankan API di port 8080. Pakai [Air](https://github.com/air-verse/air) supaya
   rebuild otomatis tiap file `.go` atau `.env` berubah, atau cukup `go run ./cmd/api`.

   ```bash
   go install github.com/air-verse/air@latest
   air
   ```

4. Jalankan frontend di port 5173. Vite mem-proxy `/api/*` ke `localhost:8080`.

   ```bash
   cd web && npm install && npm run dev
   ```

5. Buka http://localhost:5173, upload dokumen, lalu mulai bertanya.

## Deploy (VPS)

Satu image Docker berisi API Go, hasil build frontend, dan Tesseract. App disajikan
dari satu origin: frontend di `/`, API di `/api`.

```bash
cp .env.example .env        # isi GROQ_API_KEY dan ADMIN_TOKEN
docker compose --profile prod up -d --build
docker compose exec ollama ollama pull bge-m3
```

- App jalan di port `8080` (ubah dengan `APP_PORT`). Taruh reverse proxy (Caddy/nginx)
  dengan HTTPS di depannya, lalu set `TRUST_PROXY=true`.
- Isi `ADMIN_TOKEN`: tanpa token, siapa pun bisa upload dan menghapus dokumen. Chat
  tetap publik dan dibatasi rate limit.
- Qdrant dan Ollama hanya di-bind ke `127.0.0.1` karena tidak punya autentikasi.
- Image tanpa OCR (±18 MB, distroless) bisa di-build dengan `docker build --target slim .`.
- Reranker: `docker compose --profile rerank up -d` lalu set
  `RERANK_URL=http://reranker`. Hanya disarankan di host dengan GPU.

### Environment variable

| Nama | Default | Keterangan |
|---|---|---|
| `GROQ_API_KEY` | – | Wajib untuk `/chat` |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model apa pun yang tersedia di akun Groq |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Provider OpenAI-compatible lain |
| `ADMIN_TOKEN` | kosong | Wajib untuk upload/hapus jika diisi |
| `OLLAMA_URL` | `http://localhost:11434` | |
| `EMBED_MODEL` | `bge-m3` | Dimensi dideteksi otomatis. Tiap model punya collection sendiri |
| `COLLECTION` | `chunks_<model>` | Nama collection Qdrant |
| `QDRANT_URL` | `http://localhost:6333` | |
| `QDRANT_API_KEY` | kosong | Jika Qdrant memakai autentikasi |
| `HYBRID_SEARCH` | `true` | `false` = dense saja (untuk perbandingan) |
| `MIN_SCORE` | `0.45` | Cosine minimum (disetel untuk bge-m3; nomic: `0.64`). `0` = mati |
| `SCORE_MARGIN` | `0.10` | Buang chunk yang lebih dari ini di bawah hit teratas. `0` = mati |
| `RERANK_URL` | kosong | URL Text Embeddings Inference untuk reranking |
| `RERANK_MIN_SCORE` | `0.1` | Skor reranker minimum |
| `OCR` | aktif jika ada | `off` untuk mematikan OCR walau Tesseract terpasang |
| `OCR_LANGS` | `ind+eng` | Bahasa Tesseract |
| `DATA_DIR` | `data` | Folder file asli yang di-upload |
| `STATIC_DIR` | kosong | Folder build frontend. Jika diisi, API pindah ke `/api` |
| `CHAT_RATE_PER_MIN` | `20` | Batas `/chat` dan `/search` per IP. `0` = mati |
| `INGEST_RATE_PER_MIN` | `5` | Batas `/ingest` per IP. `0` = mati |
| `TRUST_PROXY` | `false` | `true` hanya di belakang reverse proxy yang mengisi `X-Forwarded-For` |
| `PORT` | `8080` | |

`.env` hanya dibaca untuk development lokal. Variable yang sudah di-set di environment
selalu menang.

## API

| Method | Path | Keterangan |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/config` | Fitur aktif: format yang diterima, perlu token admin atau tidak |
| `POST` | `/ingest` | Multipart field `file` (maks 20 MB) → `202 {job_id, document_id}`, `409` jika duplikat. Admin |
| `GET` | `/ingest/{job}` | Status: `queued`, `extracting`, `embedding` (`done`/`total`), `done`, atau `error` |
| `POST` | `/chat` | `{question, document_id?, history?}` → stream SSE |
| `POST` | `/search` | Retrieval saja, tanpa LLM: `{question, document_id?, raw?}` → `{sources, overview}` |
| `GET` | `/documents` | Daftar dokumen, terbaru dulu |
| `GET` | `/documents/{id}/file` | File asli |
| `DELETE` | `/documents/{id}` | Hapus dokumen, chunk, dan file. Admin |

Event SSE dari `/chat`:

```
event: token    data: {"text":"..."}     (berulang)
event: sources  data: [{n, document_id, filename, kind, page, section_title, score, text}, ...]
event: done     data: {}
event: error    data: {"error":"..."}    (pengganti done jika gagal di tengah stream)
```

## Evaluasi

```bash
go run ./cmd/eval -api http://localhost:8080        # hit@1, hit@5, MRR per tipe pertanyaan
go run ./cmd/eval -tune                              # tanpa filter skor, sarankan MIN_SCORE
```

Eval meng-upload `eval/corpus` ke server yang dituju. Jalankan pada instance
terpisah dengan `COLLECTION=eval` supaya dokumen uji tidak tercampur dengan dokumen
asli.

## Struktur project

```
rag-chatbot/
├── cmd/
│   ├── api/               # entrypoint, wiring, loader .env
│   └── eval/              # evaluasi retrieval
├── internal/
│   ├── handler/           # HTTP: ingest async, chat (SSE), search, dokumen
│   ├── extract/           # deteksi format, PDF/DOCX/PPTX/MD/TXT/gambar → bagian
│   ├── pdftext/           # teks PDF dari posisi glyph
│   ├── ocr/               # Tesseract + pdftoppm
│   ├── rag/               # chunking, prompt, fusi RRF, filter skor, sitasi
│   ├── sparse/            # vektor BM25
│   ├── vector/            # client REST Qdrant (dense, sparse, hybrid)
│   ├── embed/             # client embedding Ollama
│   ├── rerank/            # client reranker TEI
│   ├── llm/               # client streaming OpenAI-compatible
│   ├── store/             # penyimpanan file asli
│   ├── ratelimit/         # token bucket per IP
│   └── server/            # produksi: API di /api + file statis SPA
├── web/                   # frontend Vite + React
├── eval/                  # korpus, pertanyaan, hasil
├── Dockerfile             # target full (OCR, default) dan slim
└── docker-compose.yml     # Qdrant, Ollama; profile prod (app) dan rerank
```

## Test

```bash
go test ./...
cd web && npm run lint && npm run build
```

Unit test memakai `httptest` dan fake in-memory, jadi tidak butuh Qdrant, Ollama, atau
API key. CI (GitHub Actions) menjalankan semuanya, plus `go test -race` dan build
kedua image Docker.

## Catatan desain

- **Ekstraksi PDF dari posisi glyph.** Banyak PDF (terutama LaTeX) tidak menggambar
  spasi, cuma menggeser posisi. Library PDF menggabungkan kata-kata menjadi satu,
  jadi teks disusun ulang: celah antar glyph jadi spasi, perubahan baseline jadi baris
  baru, dan kata yang terpotong tanda hubung disambung lagi.
- **Hybrid dengan skor yang tetap bermakna.** Urutan memakai RRF, tetapi filter tetap
  memakai cosine dense, sehingga `MIN_SCORE` punya arti yang sama di mode dense dan
  hybrid. Percobaan pertama, yang hanya memakai BM25 untuk menambah kandidat, tidak
  mengubah hasil sama sekali (lihat `eval/RESULTS.md`).
- **Pertanyaan tentang dokumen secara keseluruhan** skornya tidak lebih tinggi dari
  pertanyaan di luar topik. Jadi saat satu dokumen dipilih dan tidak ada yang cocok,
  bagian awal dokumen dikirim, dan model yang memutuskan bisa menjawab atau tidak.
- **Multi-turn tanpa state di server.** Client mengirim `history` (maksimal 6 pesan).
  Retrieval dijalankan dengan pertanyaan saja dan dengan pertanyaan sebelumnya +
  pertanyaan sekarang, lalu peringkat terbaik per chunk diambil.
- **Satu collection per model embedding.** Dokumen dibedakan lewat payload
  `document_id` (ber-index). Ganti model tidak mencampur vektor yang dimensinya beda.
- **Ukuran chunk dihitung dalam kata** (±0,75 kata per token), supaya tidak butuh
  tokenizer.
