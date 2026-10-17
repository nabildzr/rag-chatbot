// Package handler exposes the HTTP API: ingest, chat and document management.
package handler

import (
	"context"
	"encoding/json"
	"net/http"
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
	mux.HandleFunc("POST /ingest", h.deps.IngestLimit.Wrap(h.ingest))
	mux.HandleFunc("POST /chat", h.deps.ChatLimit.Wrap(h.chat))
	mux.HandleFunc("GET /documents", h.listDocuments)
	mux.HandleFunc("DELETE /documents/{id}", h.deleteDocument)
	return mux
}

func (h *Handler) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
