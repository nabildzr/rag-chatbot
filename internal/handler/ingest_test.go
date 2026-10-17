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
	"sync"
	"testing"
	"time"

	"github.com/nabildzr/rag-chatbot/internal/extract"
	"github.com/nabildzr/rag-chatbot/internal/sparse"
	"github.com/nabildzr/rag-chatbot/internal/store"
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

func (fakeEmbedder) Dimension(context.Context) (int, error) { return 2, nil }

// fakeStore is an in-memory VectorStore with equality-only filters. Ingest
// runs in the background, so access is locked.
type fakeStore struct {
	mu        sync.Mutex
	exists    bool
	points    []vector.Point
	indexes   []string
	upsertErr error
	deleted   *vector.Filter
}

func (s *fakeStore) CollectionExists(context.Context, string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.exists, nil
}

func (s *fakeStore) EnsureCollection(context.Context, string, int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.exists = true
	return nil
}

func (s *fakeStore) CreatePayloadIndex(_ context.Context, _, field, _ string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.indexes = append(s.indexes, field)
	return nil
}

func (s *fakeStore) Upsert(_ context.Context, _ string, p []vector.Point) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.upsertErr != nil {
		return s.upsertErr
	}
	s.points = append(s.points, p...)
	return nil
}

func (s *fakeStore) Search(context.Context, string, []float32, int, *vector.Filter, string) ([]vector.ScoredPoint, error) {
	return nil, nil
}

// The fake behaves like a collection created before hybrid search.
func (s *fakeStore) Hybrid(context.Context, string) (bool, error) { return false, nil }

func (s *fakeStore) SparseSearch(context.Context, string, sparse.Vector, int, *vector.Filter) ([]vector.ScoredPoint, error) {
	return nil, errors.New("not hybrid")
}

func (s *fakeStore) Scroll(_ context.Context, _ string, f *vector.Filter, limit int, _ any) ([]vector.Record, any, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []vector.Record
	for _, p := range s.points {
		if matches(p.Payload, f) && len(out) < limit {
			out = append(out, vector.Record{ID: p.ID, Payload: p.Payload})
		}
	}
	return out, nil, nil
}

func (s *fakeStore) SetPayload(_ context.Context, _ string, payload map[string]any, f *vector.Filter) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, p := range s.points {
		if matches(p.Payload, f) {
			for k, v := range payload {
				p.Payload[k] = v
			}
		}
	}
	return nil
}

func (s *fakeStore) DeleteByFilter(_ context.Context, _ string, f *vector.Filter) error {
	s.mu.Lock()
	defer s.mu.Unlock()
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

func (s *fakeStore) snapshot() []vector.Point {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]vector.Point(nil), s.points...)
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
	for _, c := range f.MustNot {
		if v, ok := payload[c.Key]; ok && fmt.Sprint(v) == fmt.Sprint(c.Match.Value) {
			return false
		}
	}
	return true
}

func twoPages(string, []byte) (*extract.Document, error) {
	return &extract.Document{Kind: extract.PDF, Sections: []extract.Section{
		{Number: 1, Text: "halo dunia"},
		{Number: 2, Title: "Bab 2", Text: "retrieval augmented generation"},
	}}, nil
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

// uploadAndWait posts a file, waits for its background job and returns the
// final job status as seen through GET /ingest/{id}.
func uploadAndWait(t *testing.T, h *Handler, filename string, content []byte) jobSnapshot {
	t.Helper()
	routes := h.Routes()
	rec := upload(t, routes, filename, content)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("upload status %d: %s", rec.Code, rec.Body)
	}
	var acc ingestAccepted
	json.NewDecoder(rec.Body).Decode(&acc)
	select {
	case <-h.jobs.get(acc.JobID).finished:
	case <-time.After(5 * time.Second):
		t.Fatal("ingest job did not finish")
	}

	rec = httptest.NewRecorder()
	routes.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/ingest/"+acc.JobID, nil))
	var snap jobSnapshot
	json.NewDecoder(rec.Body).Decode(&snap)
	return snap
}

func TestIngestIndexesChunksInBackground(t *testing.T) {
	vs := &fakeStore{}
	files, _ := store.New(t.TempDir())
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: vs, Extract: twoPages, Files: files})

	snap := uploadAndWait(t, h, "notes.pdf", []byte("%PDF-1.7 fake"))
	if snap.Status != "done" || snap.Done != 2 || snap.Total != 2 || snap.Document == nil {
		t.Fatalf("bad job: %+v", snap)
	}
	if d := snap.Document; d.Chunks != 2 || d.Pages != 2 || d.Kind != "pdf" || d.Filename != "notes.pdf" {
		t.Fatalf("bad document: %+v", d)
	}
	points := vs.snapshot()
	if len(points) != 2 {
		t.Fatalf("stored %d points, want 2", len(points))
	}
	p := points[1].Payload
	if p["document_id"] != snap.DocumentID || p["page"] != 2 || p["section_title"] != "Bab 2" || p["ready"] != true || p["sha256"] == "" {
		t.Fatalf("bad payload: %v", p)
	}
	if _, err := files.Path(snap.DocumentID); err != nil {
		t.Fatalf("original file not stored: %v", err)
	}
}

func TestIngestRejectsDuplicate(t *testing.T) {
	vs := &fakeStore{}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: vs, Extract: twoPages})
	first := uploadAndWait(t, h, "a.pdf", []byte("%PDF-1.7 same"))

	rec := upload(t, h.Routes(), "renamed.pdf", []byte("%PDF-1.7 same"))
	var body map[string]string
	json.NewDecoder(rec.Body).Decode(&body)
	if rec.Code != http.StatusConflict || body["document_id"] != first.DocumentID {
		t.Fatalf("status %d body %v", rec.Code, body)
	}
}

func TestIngestRejectsUnsupported(t *testing.T) {
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: &fakeStore{}}).Routes()
	for name, data := range map[string][]byte{"evil.pdf": []byte("not a pdf"), "pic.png": {0x89, 'P', 'N', 'G'}} {
		if rec := upload(t, h, name, data); rec.Code != http.StatusUnsupportedMediaType {
			t.Errorf("%s: status %d, want 415", name, rec.Code)
		}
	}
}

func TestIngestEmptyTextFailsJob(t *testing.T) {
	noText := func(string, []byte) (*extract.Document, error) { return &extract.Document{Kind: extract.PDF}, nil }
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: &fakeStore{}, Extract: noText})
	snap := uploadAndWait(t, h, "scan.pdf", []byte("%PDF-1.4"))
	if snap.Status != "error" || !strings.Contains(snap.Error, "tidak ada teks") {
		t.Fatalf("bad job: %+v", snap)
	}
}

func TestIngestCleansUpOnFailure(t *testing.T) {
	vs := &fakeStore{upsertErr: errors.New("qdrant down")}
	files, _ := store.New(t.TempDir())
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: vs, Extract: twoPages, Files: files})

	snap := uploadAndWait(t, h, "a.pdf", []byte("%PDF-1.7"))
	if snap.Status != "error" || strings.Contains(snap.Error, "qdrant") {
		t.Fatalf("infrastructure errors must not leak: %+v", snap)
	}
	vs.mu.Lock()
	deleted := vs.deleted
	vs.mu.Unlock()
	if deleted == nil || deleted.Must[0].Key != "document_id" {
		t.Fatalf("expected cleanup by document_id, got %+v", deleted)
	}
	if _, err := files.Path(snap.DocumentID); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("file should be removed: %v", err)
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

func TestUnknownJob(t *testing.T) {
	rec := httptest.NewRecorder()
	New(Deps{}).Routes().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/ingest/nope", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status %d", rec.Code)
	}
}
