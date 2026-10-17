package vector

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"testing"
)

func TestEnsureCollectionCreatesWhenMissing(t *testing.T) {
	var created map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/collections/docs/exists":
			w.Write([]byte(`{"result":{"exists":false},"status":"ok"}`))
		case r.Method == http.MethodPut && r.URL.Path == "/collections/docs":
			json.NewDecoder(r.Body).Decode(&created)
			w.Write([]byte(`{"result":true,"status":"ok"}`))
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()

	if err := New(srv.URL, "").EnsureCollection(context.Background(), "docs", 768); err != nil {
		t.Fatal(err)
	}
	vectors := created["vectors"].(map[string]any)
	if vectors["size"].(float64) != 768 || vectors["distance"] != "Cosine" {
		t.Fatalf("bad create body: %v", created)
	}
}

func TestUpsertAndSearch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPut && r.URL.Path == "/collections/docs/points":
			if r.URL.Query().Get("wait") != "true" {
				t.Error("upsert should wait")
			}
			var body struct{ Points []Point }
			json.NewDecoder(r.Body).Decode(&body)
			if len(body.Points) != 1 || body.Points[0].Payload["text"] != "hello" {
				t.Errorf("bad points: %+v", body.Points)
			}
			w.Write([]byte(`{"result":{"status":"completed"},"status":"ok"}`))
		case r.Method == http.MethodPost && r.URL.Path == "/collections/docs/points/query":
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if body["limit"].(float64) != 5 || body["filter"] == nil {
				t.Errorf("bad query body: %v", body)
			}
			w.Write([]byte(`{"result":{"points":[{"id":"abc","score":0.9,"payload":{"text":"hello"}}]},"status":"ok"}`))
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()

	c := New(srv.URL, "")
	ctx := context.Background()
	err := c.Upsert(ctx, "docs", []Point{{ID: NewID(), Vector: []float32{0.1, 0.2}, Payload: map[string]any{"text": "hello"}}})
	if err != nil {
		t.Fatal(err)
	}
	hits, err := c.Search(ctx, "docs", []float32{0.1, 0.2}, 5, MatchField("document_id", "d1"))
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].Score != 0.9 || hits[0].Payload["text"] != "hello" {
		t.Fatalf("bad hits: %+v", hits)
	}
}

func TestErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"status":{"error":"Not found"}}`, http.StatusNotFound)
	}))
	defer srv.Close()

	if _, err := New(srv.URL, "").Search(context.Background(), "x", []float32{1}, 1, nil); err == nil {
		t.Fatal("expected error")
	}
}

func TestNewIDIsUUIDv4(t *testing.T) {
	if id := NewID(); !regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`).MatchString(id) {
		t.Fatalf("not a UUIDv4: %s", id)
	}
}
