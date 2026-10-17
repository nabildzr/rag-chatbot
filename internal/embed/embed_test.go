package embed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func fakeOllama(t *testing.T, gotInputs *[]string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/embed" {
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
		var body struct {
			Model string
			Input []string
		}
		json.NewDecoder(r.Body).Decode(&body)
		if body.Model != "nomic-embed-text" {
			t.Errorf("bad model %q", body.Model)
		}
		*gotInputs = body.Input
		embs := make([][]float32, len(body.Input))
		for i := range embs {
			embs[i] = []float32{float32(i), 0.5}
		}
		json.NewEncoder(w).Encode(map[string]any{"embeddings": embs})
	}))
}

func TestEmbedDocumentsAddsPrefix(t *testing.T) {
	var inputs []string
	srv := fakeOllama(t, &inputs)
	defer srv.Close()

	vecs, err := New(srv.URL, "nomic-embed-text").EmbedDocuments(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 2 || vecs[1][0] != 1 {
		t.Fatalf("bad vectors: %v", vecs)
	}
	if inputs[0] != "search_document: a" || inputs[1] != "search_document: b" {
		t.Fatalf("bad inputs: %v", inputs)
	}
}

func TestEmbedQueryAddsPrefix(t *testing.T) {
	var inputs []string
	srv := fakeOllama(t, &inputs)
	defer srv.Close()

	vec, err := New(srv.URL, "nomic-embed-text").EmbedQuery(context.Background(), "apa itu RAG?")
	if err != nil {
		t.Fatal(err)
	}
	if len(vec) != 2 || inputs[0] != "search_query: apa itu RAG?" {
		t.Fatalf("bad result: vec=%v inputs=%v", vec, inputs)
	}
}

func TestEmbedErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"model not found"}`, http.StatusNotFound)
	}))
	defer srv.Close()

	if _, err := New(srv.URL, "nomic-embed-text").Embed(context.Background(), []string{"x"}); err == nil {
		t.Fatal("expected error")
	}
}
