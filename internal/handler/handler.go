// Package handler exposes the HTTP API: ingest, chat and document management.
package handler

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"net/http"
	"strings"
	"sync/atomic"

	"github.com/nabildzr/rag-chatbot/internal/llm"
	"github.com/nabildzr/rag-chatbot/internal/rag"
	"github.com/nabildzr/rag-chatbot/internal/ratelimit"
	"github.com/nabildzr/rag-chatbot/internal/vector"
)

// Collection holds the chunks of every document; document_id in the payload
// tells them apart.
const Collection = "chunks"

type Embedder interface {
	EmbedDocuments(ctx context.Context, chunks []string) ([][]float32, error)
	EmbedQuery(ctx context.Context, question string) ([]float32, error)
}

type VectorStore interface {
	CollectionExists(ctx context.Context, name string) (bool, error)
	EnsureCollection(ctx context.Context, name string, vectorSize int) error
	CreatePayloadIndex(ctx context.Context, collection, field, schema string) error
	Scroll(ctx context.Context, collection string, filter *vector.Filter, limit int, offset any) ([]vector.Record, any, error)
	Upsert(ctx context.Context, collection string, points []vector.Point) error
	Search(ctx context.Context, collection string, v []float32, limit int, filter *vector.Filter) ([]vector.ScoredPoint, error)
	DeleteByFilter(ctx context.Context, collection string, filter *vector.Filter) error
}

type LLM interface {
	Stream(ctx context.Context, messages []llm.Message, onToken func(string) error) error
}

// ExtractFunc turns raw PDF bytes into page texts.
type ExtractFunc func(data []byte) ([]rag.Page, error)

type Deps struct {
	Embedder   Embedder
	Vectors    VectorStore
	LLM        LLM
	ExtractPDF ExtractFunc
	VectorSize int

	// Retrieval filter for /chat; see rag.SelectSources. Zero disables each.
	MinScore    float32
	ScoreMargin float32

	// Per-client rate limits; nil means unlimited.
	ChatLimit   *ratelimit.Limiter
	IngestLimit *ratelimit.Limiter

	// AdminToken, when set, is required as "Authorization: Bearer <token>"
	// for endpoints that change data (upload, delete). Chat stays public.
	AdminToken string
}

type Handler struct {
	deps    Deps
	indexed atomic.Bool // payload indexes created
}

func New(deps Deps) *Handler {
	return &Handler{deps: deps}
}

func (h *Handler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", h.health)
	mux.HandleFunc("GET /config", h.config)
	mux.HandleFunc("POST /ingest", h.requireAdmin(h.deps.IngestLimit.Wrap(h.ingest)))
	mux.HandleFunc("POST /chat", h.deps.ChatLimit.Wrap(h.chat))
	mux.HandleFunc("GET /documents", h.listDocuments)
	mux.HandleFunc("DELETE /documents/{id}", h.requireAdmin(h.deleteDocument))
	return mux
}

func (h *Handler) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// config tells the frontend which features are on, so it can hide controls
// the visitor cannot use.
func (h *Handler) config(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"admin_required": h.deps.AdminToken != "",
	})
}

// requireAdmin checks the bearer token when AdminToken is configured.
func (h *Handler) requireAdmin(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if h.deps.AdminToken != "" {
			got, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
			if !ok || subtle.ConstantTimeCompare([]byte(got), []byte(h.deps.AdminToken)) != 1 {
				w.Header().Set("WWW-Authenticate", `Bearer realm="rag-chatbot"`)
				writeError(w, http.StatusUnauthorized, "token admin salah atau belum diisi")
				return
			}
		}
		next(w, r)
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
