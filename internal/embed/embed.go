// Package embed is an Ollama embeddings client. The default model is bge-m3,
// which is multilingual; nomic-embed-text also works but is English-centric.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

const DefaultModel = "bge-m3"

type Client struct {
	baseURL string
	model   string
	http    *http.Client

	// Task prefixes the model was trained with, if any.
	docPrefix, queryPrefix string

	mu  sync.Mutex
	dim int // detected on first use
}

// New creates a client for Ollama, e.g. New("http://localhost:11434", "bge-m3").
func New(baseURL, model string) *Client {
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		http:    &http.Client{Timeout: 2 * time.Minute},
	}
	// nomic-embed-text is trained with task prefixes and retrieves noticeably
	// worse without them; bge-m3 and most others take raw text.
	if strings.HasPrefix(model, "nomic-embed") {
		c.docPrefix, c.queryPrefix = "search_document: ", "search_query: "
	}
	return c
}

// Dimension returns the model's vector size, embedding a probe text once.
func (c *Client) Dimension(ctx context.Context) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dim > 0 {
		return c.dim, nil
	}
	vecs, err := c.Embed(ctx, []string{"dimension probe"})
	if err != nil {
		return 0, err
	}
	c.dim = len(vecs[0])
	return c.dim, nil
}

// CollectionName derives a Qdrant collection name from the model, so
// switching models never mixes vectors of different sizes or meaning.
func CollectionName(model string) string {
	name := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			return r
		}
		if r >= 'A' && r <= 'Z' {
			return r + ('a' - 'A')
		}
		return '_'
	}, model)
	return "chunks_" + name
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
	return c.Embed(ctx, withPrefix(c.docPrefix, chunks))
}

// EmbedQuery embeds a user question, adding the query task prefix.
func (c *Client) EmbedQuery(ctx context.Context, question string) ([]float32, error) {
	vecs, err := c.Embed(ctx, []string{c.queryPrefix + question})
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
