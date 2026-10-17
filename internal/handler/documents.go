package handler

import (
	"context"
	"log"
	"net/http"
	"sort"

	"github.com/nabildzr/rag-chatbot/internal/vector"
)

type document struct {
	DocumentID string `json:"document_id"`
	Filename   string `json:"filename"`
	Kind       string `json:"kind"`
	Pages      int    `json:"pages"`
	Chunks     int    `json:"chunks"`
	CreatedAt  string `json:"created_at"`
}

// ensureCollection creates the chunks collection and its payload indexes.
// Index creation is idempotent, so it runs until it first succeeds.
func (h *Handler) ensureCollection(ctx context.Context) error {
	dim, err := h.deps.Embedder.Dimension(ctx)
	if err != nil {
		return err
	}
	if err := h.deps.Vectors.EnsureCollection(ctx, h.collection, dim); err != nil {
		return err
	}
	if h.indexed.Load() {
		return nil
	}
	hybrid, err := h.deps.Vectors.Hybrid(ctx, h.collection)
	if err != nil {
		return err
	}
	if !hybrid {
		log.Printf("collection %s has no keyword index (created before hybrid search); using dense search only", h.collection)
	}
	h.named.Store(hybrid)
	h.hybrid.Store(hybrid && !h.deps.DenseOnly)
	for field, schema := range map[string]string{
		"document_id": "keyword", "chunk_index": "integer", "sha256": "keyword", "ready": "bool",
	} {
		if err := h.deps.Vectors.CreatePayloadIndex(ctx, h.collection, field, schema); err != nil {
			return err
		}
	}
	h.indexed.Store(true)
	return nil
}

// listDocuments returns every ingested document, newest first. Each document
// is represented by its first chunk, which carries the document-level fields.
func (h *Handler) listDocuments(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	exists, err := h.deps.Vectors.CollectionExists(ctx, h.collection)
	if err != nil {
		log.Printf("documents: %v", err)
		writeError(w, http.StatusBadGateway, "vector store unavailable")
		return
	}
	docs := []document{}
	if !exists {
		writeJSON(w, http.StatusOK, docs)
		return
	}

	filter := vector.MatchField("chunk_index", 0).Not("ready", false)
	var offset any
	for {
		records, next, err := h.deps.Vectors.Scroll(ctx, h.collection, filter, 256, offset)
		if err != nil {
			log.Printf("documents: %v", err)
			writeError(w, http.StatusBadGateway, "vector store unavailable")
			return
		}
		for _, rec := range records {
			p := rec.Payload
			docs = append(docs, document{
				DocumentID: str(p["document_id"]),
				Filename:   str(p["filename"]),
				Kind:       kindOf(p),
				Pages:      num(p["doc_pages"]),
				Chunks:     num(p["doc_chunks"]),
				CreatedAt:  str(p["created_at"]),
			})
		}
		if next == nil {
			break
		}
		offset = next
	}

	// RFC 3339 UTC timestamps sort correctly as strings.
	sort.Slice(docs, func(i, j int) bool { return docs[i].CreatedAt > docs[j].CreatedAt })
	writeJSON(w, http.StatusOK, docs)
}

// deleteDocument removes all chunks of a document.
func (h *Handler) deleteDocument(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	id := r.PathValue("id")
	filter := vector.MatchField("document_id", id)

	exists, err := h.deps.Vectors.CollectionExists(ctx, h.collection)
	if err == nil && exists {
		var records []vector.Record
		records, _, err = h.deps.Vectors.Scroll(ctx, h.collection, filter, 1, nil)
		exists = len(records) > 0
	}
	if err != nil {
		log.Printf("delete %s: %v", id, err)
		writeError(w, http.StatusBadGateway, "vector store unavailable")
		return
	}
	if !exists {
		writeError(w, http.StatusNotFound, "document not found")
		return
	}

	if err := h.deps.Vectors.DeleteByFilter(ctx, h.collection, filter); err != nil {
		log.Printf("delete %s: %v", id, err)
		writeError(w, http.StatusBadGateway, "delete failed")
		return
	}
	if h.deps.Files != nil {
		if err := h.deps.Files.Delete(id); err != nil {
			log.Printf("delete %s: file: %v", id, err)
		}
	}
	w.WriteHeader(http.StatusNoContent)
}

// kindOf reads the file type; documents indexed before it was stored are PDFs.
func kindOf(p map[string]any) string {
	if k := str(p["kind"]); k != "" {
		return k
	}
	return "pdf"
}
