package handler

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nabildzr/rag-chatbot/internal/sparse"
	"github.com/nabildzr/rag-chatbot/internal/vector"
)

// hybridStore is a hybrid collection: dense hits come from Search, keyword
// hits from SparseSearch (with their dense vectors).
type hybridStore struct {
	fakeStore
	dense, keyword []vector.ScoredPoint
	usedVector     string
}

func (s *hybridStore) Hybrid(context.Context, string) (bool, error) { return true, nil }

func (s *hybridStore) Search(_ context.Context, _ string, _ []float32, _ int, _ *vector.Filter, using string) ([]vector.ScoredPoint, error) {
	s.usedVector = using
	return s.dense, nil
}

func (s *hybridStore) SparseSearch(context.Context, string, sparse.Vector, int, *vector.Filter) ([]vector.ScoredPoint, error) {
	return s.keyword, nil
}

func point(id, file, text string, score float32, dense []float32) vector.ScoredPoint {
	return vector.ScoredPoint{ID: id, Score: score, Dense: dense, Payload: map[string]any{
		"document_id": "d", "filename": file, "page": float64(1), "chunk_index": float64(0), "text": text,
	}}
}

func TestHybridKeepsExactKeywordMatchBelowThreshold(t *testing.T) {
	store := &hybridStore{
		fakeStore: fakeStore{exists: true},
		dense: []vector.ScoredPoint{
			point("a", "sejarah.md", "Borobudur dibangun abad ke-8.", 0.42, nil),
		},
		// Query vector from fakeEmbedder is [1 0]; cosine with [0.4 0.9165] is 0.40.
		keyword: []vector.ScoredPoint{
			point("b", "kunjungan.md", "Prosesi Waisak berangkat dari Candi Mendut.", 7.1, []float32{0.4, 0.9165}),
		},
	}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: store, MinScore: 0.45, ScoreMargin: 0.1})

	rec := httptest.NewRecorder()
	h.Routes().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/search", strings.NewReader(`{"question":"Mendut"}`)))
	var res retrieval
	json.NewDecoder(rec.Body).Decode(&res)
	if len(res.Sources) != 1 || res.Sources[0].Filename != "kunjungan.md" {
		t.Fatalf("want only the exact keyword match, got %+v", res.Sources)
	}
	if s := res.Sources[0].Score; s < 0.39 || s > 0.41 {
		t.Fatalf("keyword hit must carry its cosine score, got %.3f", s)
	}
	if store.usedVector != vector.DenseName {
		t.Fatalf("dense search on a hybrid collection must name the vector, got %q", store.usedVector)
	}
}

type fakeReranker struct {
	scores map[string]float32 // by passage text
	err    error
}

func (r fakeReranker) Rerank(_ context.Context, _ string, texts []string) ([]float32, error) {
	if r.err != nil {
		return nil, r.err
	}
	out := make([]float32, len(texts))
	for i, t := range texts {
		out[i] = r.scores[t]
	}
	return out, nil
}

func searchFiles(t *testing.T, h *Handler, question string) []string {
	t.Helper()
	rec := httptest.NewRecorder()
	h.Routes().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/search", strings.NewReader(`{"question":"`+question+`"}`)))
	var res retrieval
	json.NewDecoder(rec.Body).Decode(&res)
	var files []string
	for _, s := range res.Sources {
		files = append(files, s.Filename)
	}
	return files
}

func TestRerankerReordersAndFilters(t *testing.T) {
	store := &hybridStore{fakeStore: fakeStore{exists: true}, dense: []vector.ScoredPoint{
		point("a", "first-by-cosine.md", "A", 0.80, nil),
		point("b", "really-relevant.md", "B", 0.70, nil),
		point("c", "irrelevant.md", "C", 0.69, nil),
	}}
	rr := fakeReranker{scores: map[string]float32{"A": 0.30, "B": 0.95, "C": 0.01}}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: store, Reranker: rr, RerankMinScore: 0.2, MinScore: 0.45})

	got := searchFiles(t, h, "pertanyaan")
	if len(got) != 2 || got[0] != "really-relevant.md" || got[1] != "first-by-cosine.md" {
		t.Fatalf("got %v", got)
	}
}

func TestRerankerFailureFallsBack(t *testing.T) {
	store := &hybridStore{fakeStore: fakeStore{exists: true}, dense: []vector.ScoredPoint{
		point("a", "doc.md", "A", 0.80, nil),
	}}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: store, Reranker: fakeReranker{err: errors.New("down")}, MinScore: 0.45})
	if got := searchFiles(t, h, "pertanyaan"); len(got) != 1 || got[0] != "doc.md" {
		t.Fatalf("got %v", got)
	}
}
