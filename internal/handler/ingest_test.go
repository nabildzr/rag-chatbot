package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nabildzr/rag-chatbot/internal/rag"
	"github.com/nabildzr/rag-chatbot/internal/vector"
)

type fakeEmbedder struct{}

func (fakeEmbedder) EmbedDocuments(_ context.Context, chunks []string) ([][]float32, error) {
	out := make([][]float32, len(chunks))
	for i := range out {
		out[i] = []float32{1, 0}
	}
	return out, nil
}

func (fakeEmbedder) EmbedQuery(context.Context, string) ([]float32, error) {
	return []float32{1, 0}, nil
}

// fakeStore is an in-memory VectorStore with equality-only filters.
type fakeStore struct {
	exists    bool
	points    []vector.Point
	indexes   []string
	upsertErr error
	deleted   *vector.Filter
}

func (s *fakeStore) CollectionExists(context.Context, string) (bool, error) { return s.exists, nil }

func (s *fakeStore) EnsureCollection(context.Context, string, int) error {
	s.exists = true
	return nil
}

func (s *fakeStore) CreatePayloadIndex(_ context.Context, _, field, _ string) error {
	s.indexes = append(s.indexes, field)
	return nil
}

func (s *fakeStore) Upsert(_ context.Context, _ string, p []vector.Point) error {
	if s.upsertErr != nil {
		return s.upsertErr
	}
	s.points = append(s.points, p...)
	return nil
}

func (s *fakeStore) Search(context.Context, string, []float32, int, *vector.Filter) ([]vector.ScoredPoint, error) {
	return nil, nil
}

func (s *fakeStore) Scroll(_ context.Context, _ string, f *vector.Filter, limit int, _ any) ([]vector.Record, any, error) {
	var out []vector.Record
	for _, p := range s.points {
		if matches(p.Payload, f) && len(out) < limit {
			out = append(out, vector.Record{ID: p.ID, Payload: p.Payload})
		}
	}
	return out, nil, nil
}

func (s *fakeStore) DeleteByFilter(_ context.Context, _ string, f *vector.Filter) error {
	s.deleted = f
	kept := s.points[:0]
	for _, p := range s.points {
		if !matches(p.Payload, f) {
			kept = append(kept, p)
		}
	}
	s.points = kept
	return nil
}

func matches(payload map[string]any, f *vector.Filter) bool {
	if f == nil {
		return true
	}
	for _, c := range f.Must {
		if fmt.Sprint(payload[c.Key]) != fmt.Sprint(c.Match.Value) {
			return false
		}
	}
	return true
}

func twoPages([]byte) ([]rag.Page, error) {
	return []rag.Page{{Number: 1, Text: "halo dunia"}, {Number: 2, Text: "retrieval augmented generation"}}, nil
}

func upload(t *testing.T, h http.Handler, filename string, content []byte) *httptest.ResponseRecorder {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, _ := mw.CreateFormFile("file", filename)
	fw.Write(content)
	mw.Close()

	req := httptest.NewRequest(http.MethodPost, "/ingest", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestIngestIndexesChunks(t *testing.T) {
	store := &fakeStore{}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: store, ExtractPDF: twoPages, VectorSize: 2}).Routes()

	rec := upload(t, h, "notes.pdf", []byte("%PDF-1.7 fake"))
	if rec.Code != http.StatusCreated {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var res ingestResponse
	json.NewDecoder(rec.Body).Decode(&res)
	if res.Chunks != 2 || res.Pages != 2 || res.Filename != "notes.pdf" || res.DocumentID == "" {
		t.Fatalf("bad response: %+v", res)
	}
	if len(store.points) != 2 {
		t.Fatalf("stored %d points, want 2", len(store.points))
	}
	p := store.points[1].Payload
	if p["document_id"] != res.DocumentID || p["page"] != 2 || p["text"] != "retrieval augmented generation" {
		t.Fatalf("bad payload: %v", p)
	}
}

func TestIngestRejectsNonPDF(t *testing.T) {
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: &fakeStore{}, ExtractPDF: twoPages}).Routes()
	if rec := upload(t, h, "evil.pdf", []byte("not a pdf")); rec.Code != http.StatusUnsupportedMediaType {
		t.Fatalf("status %d, want 415", rec.Code)
	}
}

func TestIngestRejectsEmptyText(t *testing.T) {
	noText := func([]byte) ([]rag.Page, error) { return nil, nil }
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: &fakeStore{}, ExtractPDF: noText}).Routes()
	if rec := upload(t, h, "scan.pdf", []byte("%PDF-1.4")); rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status %d, want 422", rec.Code)
	}
}

func TestIngestCleansUpOnFailure(t *testing.T) {
	store := &fakeStore{upsertErr: errors.New("qdrant down")}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: store, ExtractPDF: twoPages}).Routes()

	rec := upload(t, h, "a.pdf", []byte("%PDF-1.7"))
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status %d, want 502", rec.Code)
	}
	if store.deleted == nil || store.deleted.Must[0].Key != "document_id" {
		t.Fatalf("expected cleanup by document_id, got %+v", store.deleted)
	}
}

func TestIngestRequiresFile(t *testing.T) {
	h := New(Deps{}).Routes()
	req := httptest.NewRequest(http.MethodPost, "/ingest", strings.NewReader(""))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400", rec.Code)
	}
}
