package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestStreamCollectsTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" || r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("bad request: %s auth=%q", r.URL.Path, r.Header.Get("Authorization"))
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["stream"] != true || body["model"] != "m" {
			t.Errorf("bad body: %v", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(`data: {"choices":[{"delta":{"role":"assistant"}}]}

data: {"choices":[{"delta":{"content":"Halo"}}]}

data: {"choices":[{"delta":{"content":" dunia"}}]}

data: [DONE]

`))
	}))
	defer srv.Close()

	var got strings.Builder
	err := New(srv.URL, "test-key", "m").Stream(context.Background(), []Message{{Role: "user", Content: "hi"}}, func(tok string) error {
		got.WriteString(tok)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.String() != "Halo dunia" {
		t.Fatalf("got %q", got.String())
	}
}

func TestStreamHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"invalid api key"}}`, http.StatusUnauthorized)
	}))
	defer srv.Close()

	err := New(srv.URL, "bad", "m").Stream(context.Background(), nil, func(string) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("want 401 error, got %v", err)
	}
}

func TestStreamStopsOnCallbackError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	stop := errors.New("client gone")
	calls := 0
	err := New(srv.URL, "k", "m").Stream(context.Background(), nil, func(string) error {
		calls++
		return stop
	})
	if !errors.Is(err, stop) || calls != 1 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}

func TestStreamRequiresKey(t *testing.T) {
	if err := New("http://unused", "", "m").Stream(context.Background(), nil, nil); !errors.Is(err, ErrNoAPIKey) {
		t.Fatalf("got %v", err)
	}
}
