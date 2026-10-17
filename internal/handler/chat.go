package handler

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/nabildzr/rag-chatbot/internal/llm"
	"github.com/nabildzr/rag-chatbot/internal/rag"
	"github.com/nabildzr/rag-chatbot/internal/vector"
)

const (
	topK           = 5
	maxQuestionLen = 2000 // characters
	notFoundAnswer = "Maaf, saya tidak menemukan informasi itu di dokumen."
)

type chatRequest struct {
	Question   string `json:"question"`
	DocumentID string `json:"document_id"` // optional; empty searches all documents
}

// chat answers a question over Server-Sent Events:
//
//	event: token    data: {"text":"..."}   (repeated)
//	event: sources  data: [ {...}, ... ]   (chunks used as citations)
//	event: done     data: {}
//	event: error    data: {"error":"..."}  (instead of done, on failure mid-stream)
//
// Validation and retrieval errors happen before the stream starts and are
// returned as plain JSON with a 4xx/5xx status.
func (h *Handler) chat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	req.Question = strings.TrimSpace(req.Question)
	if req.Question == "" {
		writeError(w, http.StatusBadRequest, "question is required")
		return
	}
	if utf8.RuneCountInString(req.Question) > maxQuestionLen {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("question exceeds %d characters", maxQuestionLen))
		return
	}

	ctx := r.Context()
	qvec, err := h.deps.Embedder.EmbedQuery(ctx, req.Question)
	if err != nil {
		log.Printf("chat: embed: %v", err)
		writeError(w, http.StatusBadGateway, "embedding failed")
		return
	}
	var filter *vector.Filter
	if req.DocumentID != "" {
		filter = vector.MatchField("document_id", req.DocumentID)
	}
	hits, err := h.deps.Vectors.Search(ctx, Collection, qvec, topK, filter)
	if err != nil {
		log.Printf("chat: search: %v", err)
		writeError(w, http.StatusBadGateway, "search failed")
		return
	}
	sources := toSources(hits)

	sse, err := newSSE(w)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}

	if len(sources) == 0 {
		sse.send("token", map[string]string{"text": notFoundAnswer})
		sse.send("sources", sources)
		sse.send("done", struct{}{})
		return
	}

	messages := []llm.Message{
		{Role: "system", Content: rag.SystemPrompt},
		{Role: "user", Content: rag.BuildUserPrompt(req.Question, sources)},
	}
	err = h.deps.LLM.Stream(ctx, messages, func(token string) error {
		return sse.send("token", map[string]string{"text": token})
	})
	if err != nil {
		if ctx.Err() == nil { // client still connected
			log.Printf("chat: llm: %v", err)
			sse.send("error", map[string]string{"error": "answer generation failed"})
		}
		return
	}
	sse.send("sources", sources)
	sse.send("done", struct{}{})
}

func toSources(hits []vector.ScoredPoint) []rag.Source {
	sources := make([]rag.Source, 0, len(hits))
	for _, hit := range hits {
		p := hit.Payload
		sources = append(sources, rag.Source{
			DocumentID: str(p["document_id"]),
			Filename:   str(p["filename"]),
			Page:       num(p["page"]),
			ChunkIndex: num(p["chunk_index"]),
			Score:      hit.Score,
			Text:       str(p["text"]),
		})
	}
	return sources
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

// num reads a payload number; JSON decoding yields float64.
func num(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	}
	return 0
}

type sseWriter struct {
	w  http.ResponseWriter
	rc *http.ResponseController
}

func newSSE(w http.ResponseWriter) (*sseWriter, error) {
	rc := http.NewResponseController(w)
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no") // disable nginx buffering
	w.WriteHeader(http.StatusOK)
	if err := rc.Flush(); err != nil {
		return nil, err
	}
	return &sseWriter{w: w, rc: rc}, nil
}

// send writes one event. data is JSON-encoded, so it never contains raw newlines.
func (s *sseWriter) send(event string, data any) error {
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, b); err != nil {
		return err
	}
	return s.rc.Flush()
}
