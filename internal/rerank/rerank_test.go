package rerank

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRerankRestoresInputOrder(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Query string
			Texts []string
		}
		json.NewDecoder(r.Body).Decode(&body)
		if r.URL.Path != "/rerank" || body.Query != "q" || len(body.Texts) != 3 {
			t.Errorf("bad request %s %+v", r.URL.Path, body)
		}
		w.Write([]byte(`[{"index":2,"score":0.9},{"index":0,"score":0.4},{"index":1,"score":0.01}]`))
	}))
	defer srv.Close()

	scores, err := New(srv.URL).Rerank(context.Background(), "q", []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}
	if scores[0] != 0.4 || scores[1] != 0.01 || scores[2] != 0.9 {
		t.Fatalf("got %v", scores)
	}
}

func TestRerankErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`[{"index":0,"score":0.4}]`)) // one score for two texts
	}))
	defer srv.Close()
	if _, err := New(srv.URL).Rerank(context.Background(), "q", []string{"a", "b"}); err == nil {
		t.Fatal("expected error for missing scores")
	}
}
