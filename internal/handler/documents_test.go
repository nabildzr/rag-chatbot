package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/rag-chatbot/rag-chatbot/internal/vector"
)

func chunkPoint(docID, filename, createdAt string, index, pages, chunks int) vector.Point {
	return vector.Point{ID: vector.NewID(), Payload: map[string]any{
		"document_id": docID, "filename": filename, "chunk_index": index,
		"created_at": createdAt, "doc_pages": pages, "doc_chunks": chunks,
	}}
}

func TestListDocumentsNewestFirst(t *testing.T) {
	store := &fakeStore{exists: true, points: []vector.Point{
		chunkPoint("old", "a.pdf", "2026-01-01T00:00:00Z", 0, 3, 2),
		chunkPoint("old", "a.pdf", "2026-01-01T00:00:00Z", 1, 3, 2),
		chunkPoint("new", "b.pdf", "2026-02-01T00:00:00Z", 0, 1, 1),
	}}
	h := New(Deps{Vectors: store}).Routes()

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/documents", nil))
	var docs []document
	json.NewDecoder(rec.Body).Decode(&docs)
	if rec.Code != http.StatusOK || len(docs) != 2 {
		t.Fatalf("status %d docs %+v", rec.Code, docs)
	}
	if docs[0].DocumentID != "new" || docs[1].Pages != 3 || docs[1].Chunks != 2 || docs[1].Filename != "a.pdf" {
		t.Fatalf("bad docs: %+v", docs)
	}
}

func TestListDocumentsNoCollection(t *testing.T) {
	h := New(Deps{Vectors: &fakeStore{}}).Routes()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/documents", nil))
	if rec.Code != http.StatusOK || rec.Body.String() != "[]\n" {
		t.Fatalf("status %d body %q", rec.Code, rec.Body)
	}
}

func TestDeleteDocument(t *testing.T) {
	store := &fakeStore{exists: true, points: []vector.Point{
		chunkPoint("d1", "a.pdf", "t", 0, 1, 2),
		chunkPoint("d1", "a.pdf", "t", 1, 1, 2),
		chunkPoint("d2", "b.pdf", "t", 0, 1, 1),
	}}
	h := New(Deps{Vectors: store}).Routes()

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodDelete, "/documents/d1", nil))
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status %d", rec.Code)
	}
	if len(store.points) != 1 || store.points[0].Payload["document_id"] != "d2" {
		t.Fatalf("wrong points left: %+v", store.points)
	}

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodDelete, "/documents/d1", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("second delete: status %d, want 404", rec.Code)
	}
}

func TestIngestStoresDocFieldsAndIndexes(t *testing.T) {
	store := &fakeStore{}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: store, ExtractPDF: twoPages}).Routes()
	upload(t, h, "notes.pdf", []byte("%PDF-1.7"))

	p := store.points[0].Payload
	if p["doc_pages"] != 2 || p["doc_chunks"] != 2 {
		t.Fatalf("bad doc fields: %v", p)
	}
	if len(store.indexes) != 2 {
		t.Fatalf("want 2 payload indexes, got %v", store.indexes)
	}
}
