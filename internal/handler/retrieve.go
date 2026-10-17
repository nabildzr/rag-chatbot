package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/nabildzr/rag-chatbot/internal/llm"
	"github.com/nabildzr/rag-chatbot/internal/rag"
	"github.com/nabildzr/rag-chatbot/internal/sparse"
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

// candidatePool is how many chunks each index (dense, keyword) contributes
// before the final ordering.
const candidatePool = 20

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

	// Learn the collection layout (and create it) before the first search.
	exists, err := h.deps.Vectors.CollectionExists(ctx, h.collection)
	if err != nil {
		return retrieval{}, fmt.Errorf("collection: %w", err)
	}
	if !exists {
		return retrieval{Sources: []rag.Source{}}, nil
	}
	if err := h.ensureCollection(ctx); err != nil {
		return retrieval{}, fmt.Errorf("collection: %w", err)
	}

	var sources []rag.Source
	minScore, margin := h.deps.MinScore, h.deps.ScoreMargin
	if raw {
		minScore, margin = 0, 0
	}
	if h.hybrid.Load() {
		cands, err := h.hybridCandidates(ctx, question, queries, filter)
		if err != nil {
			return retrieval{}, err
		}
		sources = rag.Fuse(cands, minScore, margin, topK)
	} else {
		var hits []vector.ScoredPoint
		using := ""
		if h.named.Load() {
			using = vector.DenseName
		}
		for _, q := range queries {
			qvec, err := h.deps.Embedder.EmbedQuery(ctx, q)
			if err != nil {
				return retrieval{}, fmt.Errorf("embed: %w", err)
			}
			found, err := h.deps.Vectors.Search(ctx, h.collection, qvec, topK, filter, using)
			if err != nil {
				return retrieval{}, fmt.Errorf("search: %w", err)
			}
			hits = append(hits, found...)
		}
		sources = rag.SelectSources(toSources(mergeHits(hits, topK)), minScore, margin)
	}
	if raw || len(sources) > 0 || docID == "" {
		return retrieval{Sources: sources}, nil
	}

	// Questions about a whole document ("what is this about?") match no chunk
	// better than off-topic ones do, so scores cannot tell them apart. With one
	// document selected, fall back to its opening chunks and let the model decide.
	sources, err = h.overviewSources(ctx, docID)
	if err != nil {
		return retrieval{}, fmt.Errorf("overview: %w", err)
	}
	return retrieval{Sources: sources, Overview: len(sources) > 0}, nil
}

// hybridCandidates runs a dense and a keyword search for every query and
// merges the results per chunk: best rank in each list, best cosine. Keyword
// hits get their cosine from the dense vector Qdrant returns with them.
func (h *Handler) hybridCandidates(ctx context.Context, question string, queries []string, filter *vector.Filter) ([]rag.Candidate, error) {
	byID := map[string]*rag.Candidate{}
	add := func(p vector.ScoredPoint, cosine float32, denseRank, kwRank int) {
		key := fmt.Sprint(p.ID)
		c, ok := byID[key]
		if !ok {
			src := toSources([]vector.ScoredPoint{p})[0]
			src.Score = 0 // p.Score may be a BM25 score; only cosine counts here
			// Exactness is judged on the question itself, not on the
			// follow-up query that also contains the previous question.
			c = &rag.Candidate{Source: src, Exact: sparse.ContainsAll(src.Text, question)}
			byID[key] = c
		}
		c.Score = max(c.Score, cosine)
		if denseRank > 0 && (c.DenseRank == 0 || denseRank < c.DenseRank) {
			c.DenseRank = denseRank
		}
		if kwRank > 0 && (c.KeywordRank == 0 || kwRank < c.KeywordRank) {
			c.KeywordRank = kwRank
		}
	}

	for _, q := range queries {
		qvec, err := h.deps.Embedder.EmbedQuery(ctx, q)
		if err != nil {
			return nil, fmt.Errorf("embed: %w", err)
		}
		dense, err := h.deps.Vectors.Search(ctx, h.collection, qvec, candidatePool, filter, vector.DenseName)
		if err != nil {
			return nil, fmt.Errorf("dense search: %w", err)
		}
		for i, p := range dense {
			add(p, p.Score, i+1, 0)
		}
		kw, err := h.deps.Vectors.SparseSearch(ctx, h.collection, sparse.Query(q), candidatePool, filter)
		if err != nil {
			return nil, fmt.Errorf("keyword search: %w", err)
		}
		for i, p := range kw {
			add(p, cosine(qvec, p.Dense), 0, i+1)
		}
	}

	out := make([]rag.Candidate, 0, len(byID))
	for _, c := range byID {
		out = append(out, *c)
	}
	return out, nil
}

func cosine(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
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
		log.Printf("search: %v", err)
		writeError(w, http.StatusBadGateway, "search failed")
		return
	}
	if res.Sources == nil {
		res.Sources = []rag.Source{}
	}
	writeJSON(w, http.StatusOK, res)
}
