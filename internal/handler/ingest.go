package handler

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"time"

	"github.com/nabildzr/rag-chatbot/internal/extract"
	"github.com/nabildzr/rag-chatbot/internal/rag"
	"github.com/nabildzr/rag-chatbot/internal/sparse"
	"github.com/nabildzr/rag-chatbot/internal/vector"
)

const (
	maxUploadBytes = 20 << 20 // 20 MB
	embedBatchSize = 32
	ingestTimeout  = 30 * time.Minute
)

type ingestAccepted struct {
	JobID      string `json:"job_id"`
	DocumentID string `json:"document_id"`
	Filename   string `json:"filename"`
}

// ingest validates an upload (multipart field "file"), stores it and queues
// indexing in the background. It answers 202 with a job ID to poll at
// GET /ingest/{job}; 409 with the existing document if the same file was
// already indexed.
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
	filename := filepath.Base(header.Filename)
	if _, err := extract.Detect(filename, data); err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err.Error())
		return
	}

	sum := sha256.Sum256(data)
	hash := hex.EncodeToString(sum[:])
	ctx := r.Context()
	if err := h.ensureCollection(ctx); err != nil {
		log.Printf("ingest %s: %v", filename, err)
		writeError(w, http.StatusBadGateway, "vector store unavailable")
		return
	}
	if existing, err := h.findByHash(ctx, hash); err != nil {
		log.Printf("ingest %s: %v", filename, err)
		writeError(w, http.StatusBadGateway, "vector store unavailable")
		return
	} else if existing != "" {
		writeJSON(w, http.StatusConflict, map[string]string{
			"error":       "file yang sama sudah pernah di-upload",
			"document_id": existing,
		})
		return
	}

	docID := vector.NewID()
	j, dup := h.jobs.start(docID, filename, hash)
	if dup != "" {
		writeJSON(w, http.StatusConflict, map[string]string{
			"error":       "file yang sama sedang diproses",
			"document_id": dup,
		})
		return
	}
	if h.deps.Files != nil {
		if err := h.deps.Files.Save(docID, filepath.Ext(filename), data); err != nil {
			h.jobs.fail(j, err)
			log.Printf("ingest %s: save: %v", filename, err)
			writeError(w, http.StatusInternalServerError, "cannot store file")
			return
		}
	}

	go h.runIngest(j, data)
	writeJSON(w, http.StatusAccepted, ingestAccepted{JobID: j.ID, DocumentID: docID, Filename: filename})
}

// runIngest extracts, chunks, embeds and stores a document. Only one runs at
// a time so a big upload cannot starve the embedding model for everyone.
func (h *Handler) runIngest(j *job, data []byte) {
	ctx, cancel := context.WithTimeout(context.Background(), ingestTimeout)
	defer cancel()

	h.ingestSlot <- struct{}{}
	defer func() { <-h.ingestSlot }()

	doc, err := h.index(ctx, j, data)
	if err != nil {
		log.Printf("ingest %s (%s): %v", j.Filename, j.DocumentID, err)
		h.cleanup(j.DocumentID)
		h.jobs.fail(j, err)
		return
	}
	h.jobs.finish(j, doc)
}

func (h *Handler) index(ctx context.Context, j *job, data []byte) (*document, error) {
	h.jobs.update(j, "extracting", 0, 0)
	doc, err := h.deps.Extract(j.Filename, data)
	if err != nil {
		return nil, userError{err}
	}

	pages := make([]rag.Page, len(doc.Sections))
	titles := map[int]string{}
	for i, s := range doc.Sections {
		pages[i] = rag.Page{Number: s.Number, Text: s.Text}
		titles[s.Number] = s.Title
	}
	chunks := rag.ChunkPages(pages, rag.DefaultChunkWords, rag.DefaultOverlapWords)
	if len(chunks) == 0 {
		return nil, userError{errors.New("tidak ada teks yang bisa dibaca (PDF hasil scan?)")}
	}

	h.jobs.update(j, "embedding", 0, len(chunks))
	createdAt := time.Now().UTC().Format(time.RFC3339)
	for start := 0; start < len(chunks); start += embedBatchSize {
		batch := chunks[start:min(start+embedBatchSize, len(chunks))]

		texts := make([]string, len(batch))
		for i, c := range batch {
			texts[i] = c.Text
		}
		vecs, err := h.deps.Embedder.EmbedDocuments(ctx, texts)
		if err != nil {
			return nil, err
		}

		points := make([]vector.Point, len(batch))
		for i, c := range batch {
			var kw *sparse.Vector
			if h.named.Load() { // the layout needs both vectors, even when keyword search is off
				v := sparse.Document(c.Text)
				kw = &v
			}
			points[i] = vector.Point{
				ID:     vector.NewID(),
				Vector: vecs[i],
				Sparse: kw,
				Payload: map[string]any{
					"document_id":   j.DocumentID,
					"filename":      j.Filename,
					"kind":          string(doc.Kind),
					"page":          c.Page,
					"section_title": titles[c.Page],
					"chunk_index":   c.Index,
					"text":          c.Text,
					"created_at":    createdAt,
					"sha256":        j.hash,
					// Hidden from search and listing until every chunk is in.
					"ready": false,
					// Document-level fields, repeated on every chunk so
					// GET /documents can read them from chunk 0.
					"doc_pages":  len(doc.Sections),
					"doc_chunks": len(chunks),
				},
			}
		}
		if err := h.deps.Vectors.Upsert(ctx, h.collection, points); err != nil {
			return nil, err
		}
		h.jobs.update(j, "embedding", start+len(batch), len(chunks))
	}

	if err := h.deps.Vectors.SetPayload(ctx, h.collection, map[string]any{"ready": true},
		vector.MatchField("document_id", j.DocumentID)); err != nil {
		return nil, err
	}
	return &document{
		DocumentID: j.DocumentID,
		Filename:   j.Filename,
		Kind:       string(doc.Kind),
		Pages:      len(doc.Sections),
		Chunks:     len(chunks),
		CreatedAt:  createdAt,
	}, nil
}

// cleanup removes a failed document's points and file so nothing is left
// half-indexed. It uses its own context: the job's may have expired.
func (h *Handler) cleanup(docID string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := h.deps.Vectors.DeleteByFilter(ctx, h.collection, vector.MatchField("document_id", docID)); err != nil {
		log.Printf("cleanup %s: %v", docID, err)
	}
	if h.deps.Files != nil {
		if err := h.deps.Files.Delete(docID); err != nil {
			log.Printf("cleanup %s: file: %v", docID, err)
		}
	}
}

// findByHash returns the ID of an indexed document with this content hash.
func (h *Handler) findByHash(ctx context.Context, hash string) (string, error) {
	records, _, err := h.deps.Vectors.Scroll(ctx, h.collection, vector.MatchField("sha256", hash), 1, nil)
	if err != nil || len(records) == 0 {
		return "", err
	}
	return str(records[0].Payload["document_id"]), nil
}

// userError marks failures caused by the file itself; their message is
// shown to the user, unlike infrastructure errors.
type userError struct{ err error }

func (e userError) Error() string { return e.err.Error() }
func (e userError) Unwrap() error { return e.err }

func (h *Handler) ingestStatus(w http.ResponseWriter, r *http.Request) {
	j := h.jobs.get(r.PathValue("id"))
	if j == nil {
		writeError(w, http.StatusNotFound, "job not found (server restarted?)")
		return
	}
	writeJSON(w, http.StatusOK, j.snapshot())
}
