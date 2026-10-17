// Package server assembles the production HTTP handler: the API under /api
// and the built frontend everywhere else.
package server

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// New returns api at the root when staticDir is empty (development, where
// Vite serves the frontend and proxies /api). Otherwise the API moves under
// /api and staticDir (the Vite build output) is served at /.
func New(api http.Handler, staticDir string) http.Handler {
	if staticDir == "" {
		return api
	}
	mux := http.NewServeMux()
	mux.Handle("/api/", http.StripPrefix("/api", api))
	mux.Handle("/", spa(staticDir))
	return mux
}

// spa serves files from dir and falls back to index.html for unknown paths,
// so client-side routes still load the app. Hashed assets are cached hard;
// index.html is not, so new deploys show up immediately.
func spa(dir string) http.Handler {
	files := http.FileServer(http.Dir(dir))
	index := filepath.Join(dir, "index.html")
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// http.Dir already rejects ".." traversal; Clean keeps the stat consistent.
		path := filepath.Join(dir, filepath.FromSlash(filepath.Clean("/"+r.URL.Path)))
		if info, err := os.Stat(path); err != nil || info.IsDir() {
			w.Header().Set("Cache-Control", "no-cache")
			http.ServeFile(w, r, index)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/assets/") {
			w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		}
		files.ServeHTTP(w, r)
	})
}
