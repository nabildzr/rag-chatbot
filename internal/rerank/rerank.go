// Package rerank scores (question, passage) pairs with a cross-encoder served
// by Hugging Face Text Embeddings Inference (POST /rerank). A cross-encoder
// reads question and passage together, so it judges relevance better than
// comparing two separately computed embeddings, at the cost of one model pass
// per passage.
package rerank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	baseURL string
	http    *http.Client
}

func New(baseURL string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 60 * time.Second},
	}
}

// Rerank returns a relevance score in [0, 1] for each text, in input order.
func (c *Client) Rerank(ctx context.Context, query string, texts []string) ([]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	b, err := json.Marshal(map[string]any{"query": query, "texts": texts, "truncate": true})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/rerank", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rerank: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("rerank: status %d: %s", resp.StatusCode, bytes.TrimSpace(raw))
	}

	// TEI answers sorted by score; put the scores back in input order.
	var ranked []struct {
		Index int     `json:"index"`
		Score float32 `json:"score"`
	}
	if err := json.Unmarshal(raw, &ranked); err != nil {
		return nil, fmt.Errorf("rerank: decode: %w", err)
	}
	scores := make([]float32, len(texts))
	seen := 0
	for _, r := range ranked {
		if r.Index < 0 || r.Index >= len(texts) {
			return nil, fmt.Errorf("rerank: index %d out of range", r.Index)
		}
		scores[r.Index] = r.Score
		seen++
	}
	if seen != len(texts) {
		return nil, fmt.Errorf("rerank: got %d scores for %d texts", seen, len(texts))
	}
	return scores, nil
}
