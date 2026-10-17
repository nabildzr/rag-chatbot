package handler

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/nabildzr/rag-chatbot/internal/llm"
	"github.com/nabildzr/rag-chatbot/internal/rag"
	"github.com/nabildzr/rag-chatbot/internal/vector"
)

const (
	topK           = 5
	maxQuestionLen = 2000 // characters
	maxHistory     = 6    // messages (3 exchanges) kept for context
	maxTurnLen     = 2000 // characters per history message
	notFoundAnswer = "Maaf, saya tidak menemukan informasi itu di dokumen."
)

type chatRequest struct {
	Question   string        `json:"question"`
	DocumentID string        `json:"document_id"` // optional; empty searches all documents
	History    []llm.Message `json:"history"`     // optional; earlier turns, oldest first
}

// cleanHistory keeps the last maxHistory user/assistant messages, truncates
// long ones and strips stale citation markers. Clients are stateless, so
// anything else they send (system roles, empty turns) is dropped.
func cleanHistory(history []llm.Message) []llm.Message {
	var out []llm.Message
	for _, m := range history {
		if m.Role != "user" && m.Role != "assistant" {
			continue
		}
		content := strings.TrimSpace(m.Content)
		if m.Role == "assistant" {
			content = rag.StripCitations(content)
		}
		if content == "" {
			continue
		}
		if r := []rune(content); len(r) > maxTurnLen {
			content = string(r[:maxTurnLen]) + "…"
		}
		out = append(out, llm.Message{Role: m.Role, Content: content})
	}
	if len(out) > maxHistory {
		out = out[len(out)-maxHistory:]
	}
	return out
}

func lastUserMessage(history []llm.Message) string {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "user" {
			return history[i].Content
		}
	}
	return ""
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

	history := cleanHistory(req.History)

	// Search with the question alone and, for follow-ups, with the previous
	// question prepended. The standalone query wins when the topic changes,
	// the combined one when the question only makes sense in context.
	queries := []string{req.Question}
	if prev := lastUserMessage(history); prev != "" {
		queries = append(queries, rag.RetrievalQuery(req.Question, prev))
	}
	var filter *vector.Filter
	if req.DocumentID != "" {
		filter = vector.MatchField("document_id", req.DocumentID)
	}

	ctx := r.Context()
	var hits []vector.ScoredPoint
	for _, q := range queries {
		qvec, err := h.deps.Embedder.EmbedQuery(ctx, q)
		if err != nil {
			log.Printf("chat: embed: %v", err)
			writeError(w, http.StatusBadGateway, "embedding failed")
			return
		}
		found, err := h.deps.Vectors.Search(ctx, Collection, qvec, topK, filter)
		if err != nil {
			log.Printf("chat: search: %v", err)
			writeError(w, http.StatusBadGateway, "search failed")
			return
		}
		hits = append(hits, found...)
	}
	sources := rag.SelectSources(toSources(mergeHits(hits, topK)), h.deps.MinScore, h.deps.ScoreMargin)

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

	messages := make([]llm.Message, 0, len(history)+2)
	messages = append(messages, llm.Message{Role: "system", Content: rag.SystemPrompt})
	messages = append(messages, history...)
	messages = append(messages, llm.Message{Role: "user", Content: rag.BuildUserPrompt(req.Question, sources)})
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

// mergeHits de-duplicates points found by several queries, keeping each
// point's best score, and returns the top k by score.
func mergeHits(hits []vector.ScoredPoint, k int) []vector.ScoredPoint {
	best := map[string]int{} // point ID -> index in out
	var out []vector.ScoredPoint
	for _, h := range hits {
		key := fmt.Sprint(h.ID)
		if i, ok := best[key]; ok {
			if h.Score > out[i].Score {
				out[i] = h
			}
			continue
		}
		best[key] = len(out)
		out = append(out, h)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if len(out) > k {
		out = out[:k]
	}
	return out
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
