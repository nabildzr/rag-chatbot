package handler

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nabildzr/rag-chatbot/internal/llm"
	"github.com/nabildzr/rag-chatbot/internal/vector"
)

type searchStore struct {
	fakeStore
	hits       []vector.ScoredPoint
	lastFilter *vector.Filter
}

func (s *searchStore) Search(_ context.Context, _ string, _ []float32, _ int, f *vector.Filter) ([]vector.ScoredPoint, error) {
	s.lastFilter = f
	return s.hits, nil
}

type fakeLLM struct {
	tokens   []string
	err      error
	messages []llm.Message
}

func (f *fakeLLM) Stream(_ context.Context, msgs []llm.Message, onToken func(string) error) error {
	f.messages = msgs
	for _, t := range f.tokens {
		if err := onToken(t); err != nil {
			return err
		}
	}
	return f.err
}

var oneHit = []vector.ScoredPoint{{
	ID:    "p1",
	Score: 0.87,
	Payload: map[string]any{
		"document_id": "d1", "filename": "notes.pdf", "page": float64(3),
		"chunk_index": float64(7), "text": "RAG = retrieval augmented generation",
	},
}}

func postChat(h http.Handler, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/chat", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestChatStreamsTokensThenSources(t *testing.T) {
	store := &searchStore{hits: oneHit}
	model := &fakeLLM{tokens: []string{"RAG adalah", " retrieval [1]"}}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: store, LLM: model}).Routes()

	rec := postChat(h, `{"question":"Apa itu RAG?","document_id":"d1"}`)
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status %d type %q", rec.Code, rec.Header().Get("Content-Type"))
	}
	body := rec.Body.String()
	want := []string{
		"event: token\ndata: {\"text\":\"RAG adalah\"}\n\n",
		"event: token\ndata: {\"text\":\" retrieval [1]\"}\n\n",
		"event: sources\ndata: [{\"document_id\":\"d1\",\"filename\":\"notes.pdf\",\"page\":3,\"chunk_index\":7,\"score\":0.87,",
		"event: done\n",
	}
	last := 0
	for _, w := range want {
		i := strings.Index(body[last:], w)
		if i < 0 {
			t.Fatalf("missing or out of order %q in:\n%s", w, body)
		}
		last += i + len(w)
	}

	if store.lastFilter == nil || store.lastFilter.Must[0].Match.Value != "d1" {
		t.Fatalf("expected document_id filter, got %+v", store.lastFilter)
	}
	if !strings.Contains(model.messages[1].Content, "[1] (notes.pdf, page 3)") {
		t.Fatalf("prompt missing numbered source: %q", model.messages[1].Content)
	}
}

func TestChatNoHitsSkipsLLM(t *testing.T) {
	model := &fakeLLM{err: errors.New("should not be called")}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: &searchStore{}, LLM: model}).Routes()

	body := postChat(h, `{"question":"apa?"}`).Body.String()
	if model.messages != nil || !strings.Contains(body, notFoundAnswer) || !strings.Contains(body, "event: done") {
		t.Fatalf("unexpected: messages=%v body=%s", model.messages, body)
	}
}

func TestChatLLMErrorSendsErrorEvent(t *testing.T) {
	model := &fakeLLM{tokens: []string{"part"}, err: errors.New("groq 500")}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: &searchStore{hits: oneHit}, LLM: model}).Routes()

	body := postChat(h, `{"question":"apa?"}`).Body.String()
	if !strings.Contains(body, "event: error") || strings.Contains(body, "event: done") {
		t.Fatalf("unexpected body: %s", body)
	}
}

func TestChatValidation(t *testing.T) {
	h := New(Deps{}).Routes()
	for _, body := range []string{`not json`, `{"question":"   "}`, `{"question":"` + strings.Repeat("a", maxQuestionLen+1) + `"}`} {
		if rec := postChat(h, body); rec.Code != http.StatusBadRequest {
			t.Errorf("body %.30q: status %d, want 400", body, rec.Code)
		}
	}
}

func TestChatBelowMinScoreSkipsLLM(t *testing.T) {
	weak := []vector.ScoredPoint{{ID: "p1", Score: 0.61, Payload: oneHit[0].Payload}}
	model := &fakeLLM{err: errors.New("should not be called")}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: &searchStore{hits: weak}, LLM: model, MinScore: 0.64}).Routes()

	body := postChat(h, `{"question":"gaji direktur?"}`).Body.String()
	if model.messages != nil || !strings.Contains(body, notFoundAnswer) || !strings.Contains(body, "event: sources\ndata: []") {
		t.Fatalf("unexpected: messages=%v body=%s", model.messages, body)
	}
}
