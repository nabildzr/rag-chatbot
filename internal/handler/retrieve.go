package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/nabildzr/rag-chatbot/internal/llm"
	"github.com/nabildzr/rag-chatbot/internal/rag"
	"github.com/nabildzr/rag-chatbot/internal/vector"
)

// decodeQuestion reads and validates a chat or search request body,
// answering 400 itself when it is invalid.
func decodeQuestion(w http.ResponseWriter, r *http.Request) (chatRequest, bool) {
	var req chatRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return req, false
	}
	req.Question = strings.TrimSpace(req.Question)
	if req.Question == "" {
		writeError(w, http.StatusBadRequest, "question is required")
		return req, false
	}
	if utf8.RuneCountInString(req.Question) > maxQuestionLen {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("question exceeds %d characters", maxQuestionLen))
		return req, false
	}
	return req, true
}

type retrieval struct {
	Sources []rag.Source `json:"sources"`
	// Overview is true when no chunk matched and Sources are the opening
	// chunks of the selected document instead.
	Overview bool `json:"overview"`
}

// retrieve finds the chunks that answer question. It is the whole
// retrieval pipeline shared by /chat and /search. raw skips the score
// filters and the overview fallback, returning plain top-k results.
func (h *Handler) retrieve(ctx context.Context, question, docID string, history []llm.Message, raw bool) (retrieval, error) {
	// Search with the question alone and, for follow-ups, with the previous
	// question prepended. The standalone query wins when the topic changes,
	// the combined one when the question only makes sense in context.
	queries := []string{question}
	if prev := lastUserMessage(history); prev != "" {
		queries = append(queries, rag.RetrievalQuery(question, prev))
	}
	// Documents still being indexed are not searchable yet.
	filter := (*vector.Filter)(nil).Not("ready", false)
	if docID != "" {
		filter = filter.And("document_id", docID)
	}

	var hits []vector.ScoredPoint
	for _, q := range queries {
		qvec, err := h.deps.Embedder.EmbedQuery(ctx, q)
		if err != nil {
			return retrieval{}, fmt.Errorf("embed: %w", err)
		}
		found, err := h.deps.Vectors.Search(ctx, h.collection, qvec, topK, filter)
		if err != nil {
			return retrieval{}, fmt.Errorf("search: %w", err)
		}
		hits = append(hits, found...)
	}
	sources := toSources(mergeHits(hits, topK))
	if raw {
		return retrieval{Sources: sources}, nil
	}
	sources = rag.SelectSources(sources, h.deps.MinScore, h.deps.ScoreMargin)
	if len(sources) > 0 || docID == "" {
		return retrieval{Sources: sources}, nil
	}

	// Questions about a whole document ("what is this about?") match no chunk
	// better than off-topic ones do, so scores cannot tell them apart. With one
	// document selected, fall back to its opening chunks and let the model decide.
	sources, err := h.overviewSources(ctx, docID)
	if err != nil {
		return retrieval{}, fmt.Errorf("overview: %w", err)
	}
	return retrieval{Sources: sources, Overview: len(sources) > 0}, nil
}

// search runs retrieval only, without the model: for debugging and the
// evaluation tool (cmd/eval).
func (h *Handler) search(w http.ResponseWriter, r *http.Request) {
	req, ok := decodeQuestion(w, r)
	if !ok {
		return
	}
	res, err := h.retrieve(r.Context(), req.Question, req.DocumentID, cleanHistory(req.History), req.Raw)
	if err != nil {
		writeError(w, http.StatusBadGateway, "search failed")
		return
	}
	if res.Sources == nil {
		res.Sources = []rag.Source{}
	}
	writeJSON(w, http.StatusOK, res)
}
