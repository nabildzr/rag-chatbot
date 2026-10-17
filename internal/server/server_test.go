package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func setup(t *testing.T) http.Handler {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "assets"), 0o755)
	os.WriteFile(filepath.Join(dir, "index.html"), []byte("<html>app</html>"), 0o644)
	os.WriteFile(filepath.Join(dir, "assets", "index-abc.js"), []byte("console.log(1)"), 0o644)

	api := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "api:"+r.Method+" "+r.URL.Path)
	})
	return New(api, dir)
}

func get(h http.Handler, method, path string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(method, path, nil))
	return rec
}

func TestAPIMountedUnderPrefix(t *testing.T) {
	if body := get(setup(t), http.MethodDelete, "/api/documents/x").Body.String(); body != "api:DELETE /documents/x" {
		t.Fatalf("got %q", body)
	}
}

func TestSPAFallbackAndCaching(t *testing.T) {
	h := setup(t)

	// ServeMux redirects "/../x" to "/x" before we see it; "/etc/passwd"
	// must then resolve inside dir and fall back to the app.
	if rec := get(h, http.MethodGet, "/../../etc/passwd"); rec.Code != http.StatusTemporaryRedirect {
		t.Errorf("traversal: code %d", rec.Code)
	}
	for _, path := range []string{"/", "/some/client/route", "/etc/passwd"} {
		rec := get(h, http.MethodGet, path)
		if !strings.Contains(rec.Body.String(), "app") || rec.Header().Get("Cache-Control") != "no-cache" {
			t.Errorf("%s: code %d body %q cache %q", path, rec.Code, rec.Body, rec.Header().Get("Cache-Control"))
		}
	}

	rec := get(h, http.MethodGet, "/assets/index-abc.js")
	if rec.Body.String() != "console.log(1)" || !strings.Contains(rec.Header().Get("Cache-Control"), "immutable") {
		t.Fatalf("asset: body %q cache %q", rec.Body, rec.Header().Get("Cache-Control"))
	}

	if rec := get(h, http.MethodPost, "/"); rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /: %d", rec.Code)
	}
}

func TestNoStaticDirReturnsAPI(t *testing.T) {
	api := http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	if h := New(api, ""); h == nil {
		t.Fatal("nil handler")
	}
}
