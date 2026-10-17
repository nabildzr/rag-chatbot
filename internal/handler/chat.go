package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"

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
	Raw        bool          `json:"raw"`         // /search only: skip score filters and overview, for tuning
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
	req, ok := decodeQuestion(w, r)
	if !ok {
		return
	}
	history := cleanHistory(req.History)

	ctx := r.Context()
	res, err := h.retrieve(ctx, req.Question, req.DocumentID, history, false)
	if err != nil {
		log.Printf("chat: %v", err)
		writeError(w, http.StatusBadGateway, "search failed")
		return
	}
	sources, overview := res.Sources, res.Overview

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

	for i := range sources {
		sources[i].N = i + 1 // the numbers the prompt shows and the answer cites
	}
	messages := make([]llm.Message, 0, len(history)+2)
	messages = append(messages, llm.Message{Role: "system", Content: rag.SystemPrompt})
	messages = append(messages, history...)
	messages = append(messages, llm.Message{Role: "user", Content: rag.BuildUserPrompt(req.Question, sources, overview)})
	var answer strings.Builder
	err = h.deps.LLM.Stream(ctx, messages, func(token string) error {
		answer.WriteString(token)
		return sse.send("token", map[string]string{"text": token})
	})
	if err != nil {
		if ctx.Err() == nil { // client still connected
			log.Printf("chat: llm: %v", err)
			sse.send("error", map[string]string{"error": "answer generation failed"})
		}
		return
	}
	sse.send("sources", citedSources(sources, answer.String()))
	sse.send("done", struct{}{})
}

// citedSources keeps only the sources the answer cites, so an answer that
// says it found nothing shows no sources, and passages the model ignored are
// not presented as evidence. Each keeps its prompt number N.
func citedSources(sources []rag.Source, answer string) []rag.Source {
	out := []rag.Source{}
	for _, n := range rag.CitedNumbers(answer) {
		if n >= 1 && n <= len(sources) {
			out = append(out, sources[n-1])
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].N < out[j].N })
	return out
}

// overviewSources returns the first rag.OverviewChunks chunks of a document.
// They carry no similarity score (0).
func (h *Handler) overviewSources(ctx context.Context, docID string) ([]rag.Source, error) {
	var sources []rag.Source
	for i := 0; i < rag.OverviewChunks; i++ {
		filter := vector.MatchField("document_id", docID).And("chunk_index", i).Not("ready", false)
		records, _, err := h.deps.Vectors.Scroll(ctx, h.collection, filter, 1, nil)
		if err != nil {
			return nil, err
		}
		for _, rec := range records {
			sources = append(sources, toSources([]vector.ScoredPoint{{ID: rec.ID, Payload: rec.Payload}})...)
		}
	}
	return sources, nil
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
			Kind:       kindOf(p),
			Title:      str(p["section_title"]),
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
