package handler

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"time"

	"github.com/nabildzr/rag-chatbot/internal/rag"
	"github.com/nabildzr/rag-chatbot/internal/vector"
)

const (
	maxUploadBytes = 20 << 20 // 20 MB
	embedBatchSize = 32
)

type ingestResponse struct {
	DocumentID string `json:"document_id"`
	Filename   string `json:"filename"`
	Pages      int    `json:"pages"`
	Chunks     int    `json:"chunks"`
}

// ingest accepts a multipart upload with field "file" and indexes the PDF:
// extract text -> chunk -> embed -> upsert into Qdrant.
func (h *Handler) ingest(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)

	file, header, err := r.FormFile("file")
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, "file exceeds 20 MB")
			return
		}
		writeError(w, http.StatusBadRequest, `multipart field "file" is required`)
		return
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		writeError(w, http.StatusBadRequest, "cannot read upload")
		return
	}
	if !bytes.HasPrefix(data, []byte("%PDF-")) {
		writeError(w, http.StatusUnsupportedMediaType, "file is not a PDF")
		return
	}

	pages, err := h.deps.ExtractPDF(data)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	chunks := rag.ChunkPages(pages, rag.DefaultChunkWords, rag.DefaultOverlapWords)
	if len(chunks) == 0 {
		writeError(w, http.StatusUnprocessableEntity, "no extractable text (scanned PDF?)")
		return
	}

	filename := filepath.Base(header.Filename)
	docID := vector.NewID()
	if err := h.indexChunks(r.Context(), docID, filename, len(pages), chunks); err != nil {
		log.Printf("ingest %s: %v", filename, err)
		writeError(w, http.StatusBadGateway, "indexing failed")
		return
	}

	writeJSON(w, http.StatusCreated, ingestResponse{
		DocumentID: docID,
		Filename:   filename,
		Pages:      len(pages),
		Chunks:     len(chunks),
	})
}

// indexChunks embeds and stores chunks in batches. On failure it removes any
// points already written so a document is never half-indexed.
func (h *Handler) indexChunks(ctx context.Context, docID, filename string, pageCount int, chunks []rag.Chunk) (err error) {
	if err := h.ensureCollection(ctx); err != nil {
		return err
	}

	defer func() {
		if err != nil {
			// Use a fresh context: the request context may be the reason we failed.
			cleanupCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			if derr := h.deps.Vectors.DeleteByFilter(cleanupCtx, Collection, vector.MatchField("document_id", docID)); derr != nil {
				log.Printf("ingest %s: cleanup failed: %v", docID, derr)
			}
		}
	}()

	createdAt := time.Now().UTC().Format(time.RFC3339)
	for start := 0; start < len(chunks); start += embedBatchSize {
		batch := chunks[start:min(start+embedBatchSize, len(chunks))]

		texts := make([]string, len(batch))
		for i, c := range batch {
			texts[i] = c.Text
		}
		vecs, err := h.deps.Embedder.EmbedDocuments(ctx, texts)
		if err != nil {
			return err
		}

		points := make([]vector.Point, len(batch))
		for i, c := range batch {
			points[i] = vector.Point{
				ID:     vector.NewID(),
				Vector: vecs[i],
				Payload: map[string]any{
					"document_id": docID,
					"filename":    filename,
					"page":        c.Page,
					"chunk_index": c.Index,
					"text":        c.Text,
					"created_at":  createdAt,
					// Document-level fields, repeated on every chunk so
					// GET /documents can read them from chunk 0.
					"doc_pages":  pageCount,
					"doc_chunks": len(chunks),
				},
			}
		}
		if err := h.deps.Vectors.Upsert(ctx, Collection, points); err != nil {
			return err
		}
	}
	return nil
}
