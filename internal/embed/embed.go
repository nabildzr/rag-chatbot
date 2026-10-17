// Package embed is an Ollama embeddings client (nomic-embed-text).
package embed

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

// Dimension is the vector size produced by nomic-embed-text.
const Dimension = 768

// nomic-embed-text is trained with task prefixes; using them improves retrieval.
const (
	documentPrefix = "search_document: "
	queryPrefix    = "search_query: "
)

type Client struct {
	baseURL string
	model   string
	http    *http.Client
}

// New creates a client for Ollama, e.g. New("http://localhost:11434", "nomic-embed-text").
func New(baseURL, model string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		http:    &http.Client{Timeout: 2 * time.Minute},
	}
}

// Embed returns one vector per input, in the same order. Inputs are sent as-is.
func (c *Client) Embed(ctx context.Context, inputs []string) ([][]float32, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	b, err := json.Marshal(map[string]any{"model": c.model, "input": inputs})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/embed", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama embed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("ollama embed: status %d: %s", resp.StatusCode, bytes.TrimSpace(raw))
	}

	var res struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, fmt.Errorf("ollama embed: decode: %w", err)
	}
	if len(res.Embeddings) != len(inputs) {
		return nil, fmt.Errorf("ollama embed: got %d embeddings for %d inputs", len(res.Embeddings), len(inputs))
	}
	return res.Embeddings, nil
}

// EmbedDocuments embeds chunks for storage, adding the document task prefix.
func (c *Client) EmbedDocuments(ctx context.Context, chunks []string) ([][]float32, error) {
	return c.Embed(ctx, withPrefix(documentPrefix, chunks))
}

// EmbedQuery embeds a user question, adding the query task prefix.
func (c *Client) EmbedQuery(ctx context.Context, question string) ([]float32, error) {
	vecs, err := c.Embed(ctx, []string{queryPrefix + question})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

func withPrefix(prefix string, texts []string) []string {
	out := make([]string, len(texts))
	for i, t := range texts {
		out[i] = prefix + t
	}
	return out
}
