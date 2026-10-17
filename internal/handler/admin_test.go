package handler

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAdminTokenGuardsMutations(t *testing.T) {
	store := &fakeStore{exists: true}
	h := New(Deps{Embedder: fakeEmbedder{}, Vectors: store, Extract: twoPages, AdminToken: "s3cret"}).Routes()

	do := func(method, path, auth string) int {
		req := httptest.NewRequest(method, path, strings.NewReader(""))
		if auth != "" {
			req.Header.Set("Authorization", auth)
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec.Code
	}

	for _, auth := range []string{"", "Bearer wrong", "s3cret"} {
		if code := do(http.MethodDelete, "/documents/x", auth); code != http.StatusUnauthorized {
			t.Errorf("DELETE with %q: %d, want 401", auth, code)
		}
		if code := do(http.MethodPost, "/ingest", auth); code != http.StatusUnauthorized {
			t.Errorf("POST /ingest with %q: %d, want 401", auth, code)
		}
	}
	// Correct token passes the guard (404: document does not exist).
	if code := do(http.MethodDelete, "/documents/x", "Bearer s3cret"); code != http.StatusNotFound {
		t.Errorf("DELETE with token: %d, want 404", code)
	}
	// Reads stay public.
	if code := do(http.MethodGet, "/documents", ""); code != http.StatusOK {
		t.Errorf("GET /documents: %d", code)
	}
}

func TestConfigReportsAdminRequired(t *testing.T) {
	for _, token := range []string{"", "x"} {
		rec := httptest.NewRecorder()
		New(Deps{AdminToken: token}).Routes().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/config", nil))
		want := `"admin_required":` + map[bool]string{true: "true", false: "false"}[token != ""]
		if !strings.Contains(rec.Body.String(), want) {
			t.Errorf("token %q: body %s, want %s", token, rec.Body, want)
		}
	}
}
