// Package llm is a Groq client (OpenAI-compatible /chat/completions, streaming).
// Any OpenAI-compatible endpoint works by changing the base URL.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const DefaultBaseURL = "https://api.groq.com/openai/v1"

var ErrNoAPIKey = errors.New("llm: API key is not set")

type Message struct {
	Role    string `json:"role"` // "system", "user" or "assistant"
	Content string `json:"content"`
}

type Client struct {
	baseURL string
	apiKey  string
	model   string
	http    *http.Client
}

func New(baseURL, apiKey, model string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		model:   model,
		// No overall timeout: streams can be long. Callers cancel via ctx.
		http: &http.Client{},
	}
}

// Stream sends messages and calls onToken for every content delta until the
// answer is complete. Returning an error from onToken stops the stream.
func (c *Client) Stream(ctx context.Context, messages []Message, onToken func(string) error) error {
	if c.apiKey == "" {
		return ErrNoAPIKey
	}

	b, err := json.Marshal(map[string]any{
		"model":       c.model,
		"messages":    messages,
		"stream":      true,
		"temperature": 0.2,
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("llm: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("llm: status %d: %s", resp.StatusCode, bytes.TrimSpace(raw))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		data, ok := strings.CutPrefix(scanner.Text(), "data:")
		if !ok {
			continue // blank separators, comments, event names
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			return nil
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("llm: decode chunk: %w", err)
		}
		if chunk.Error != nil {
			return fmt.Errorf("llm: %s", chunk.Error.Message)
		}
		for _, ch := range chunk.Choices {
			if ch.Delta.Content == "" {
				continue
			}
			if err := onToken(ch.Delta.Content); err != nil {
				return err
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("llm: read stream: %w", err)
	}
	return errors.New("llm: stream ended without [DONE]")
}
