package handler

import (
	"context"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

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

// documentFile serves the original upload, inline, under its own name.
// Text formats go out as plain text so a browser never renders them as HTML.
// Everything but PDFs also gets a sandbox CSP; Chrome refuses to show PDFs
// under one, and its PDF viewer runs isolated anyway.
func (h *Handler) documentFile(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if h.deps.Files == nil {
		writeError(w, http.StatusNotFound, "file storage is off")
		return
	}
	path, err := h.deps.Files.Path(id)
	if err != nil {
		writeError(w, http.StatusNotFound, "file not found")
		return
	}
	records, _, err := h.deps.Vectors.Scroll(r.Context(), h.collection, vector.MatchField("document_id", id).And("chunk_index", 0), 1, nil)
	if err != nil || len(records) == 0 {
		writeError(w, http.StatusNotFound, "document not found")
		return
	}
	name := str(records[0].Payload["filename"])

	f, err := os.Open(path)
	if err != nil {
		writeError(w, http.StatusNotFound, "file not found")
		return
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "cannot read file")
		return
	}

	ctype := mime.TypeByExtension(strings.ToLower(filepath.Ext(path)))
	switch kindOf(records[0].Payload) {
	case "txt", "md":
		ctype = "text/plain; charset=utf-8"
	}
	if ctype == "" {
		ctype = "application/octet-stream"
	}
	w.Header().Set("Content-Type", ctype)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	if kindOf(records[0].Payload) != "pdf" {
		w.Header().Set("Content-Security-Policy", "sandbox")
	}
	w.Header().Set("Content-Disposition", mime.FormatMediaType("inline", map[string]string{"filename": name}))
	http.ServeContent(w, r, name, info.ModTime(), f)
}
