// Package vector is a Qdrant REST client (create collection, upsert, search).
package vector

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type Client struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

// New creates a client for the Qdrant REST API, e.g. "http://localhost:6333".
// apiKey may be empty when Qdrant runs without authentication.
func New(baseURL, apiKey string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		http:    &http.Client{Timeout: 30 * time.Second},
	}
}

type Point struct {
	ID      string         `json:"id"` // UUID; see NewID
	Vector  []float32      `json:"vector"`
	Payload map[string]any `json:"payload,omitempty"`
}

type ScoredPoint struct {
	ID      any            `json:"id"` // string UUID or uint64, as stored
	Score   float32        `json:"score"`
	Payload map[string]any `json:"payload"`
}

// Filter mirrors Qdrant's filter object with equality conditions.
type Filter struct {
	Must    []Condition `json:"must,omitempty"`
	MustNot []Condition `json:"must_not,omitempty"`
}

// And returns a copy of f with an extra required condition; f may be nil.
func (f *Filter) And(key string, value any) *Filter {
	out := &Filter{}
	if f != nil {
		out.Must = append(out.Must, f.Must...)
		out.MustNot = append(out.MustNot, f.MustNot...)
	}
	out.Must = append(out.Must, Condition{Key: key, Match: MatchValue{Value: value}})
	return out
}

// Not returns a copy of f that also excludes payload[key] == value. Points
// without the key are kept, which makes it safe for fields added later.
func (f *Filter) Not(key string, value any) *Filter {
	out := &Filter{}
	if f != nil {
		out.Must = append(out.Must, f.Must...)
		out.MustNot = append(out.MustNot, f.MustNot...)
	}
	out.MustNot = append(out.MustNot, Condition{Key: key, Match: MatchValue{Value: value}})
	return out
}

type Condition struct {
	Key   string     `json:"key"`
	Match MatchValue `json:"match"`
}

type MatchValue struct {
	Value any `json:"value"`
}

// MatchField builds a filter that requires payload[key] == value.
func MatchField(key string, value any) *Filter {
	return &Filter{Must: []Condition{{Key: key, Match: MatchValue{Value: value}}}}
}

// CollectionExists reports whether the named collection exists.
func (c *Client) CollectionExists(ctx context.Context, name string) (bool, error) {
	var res struct {
		Exists bool `json:"exists"`
	}
	err := c.do(ctx, http.MethodGet, "/collections/"+url.PathEscape(name)+"/exists", nil, &res)
	return res.Exists, err
}

// CreateCollection creates a collection with cosine distance and the given vector size.
func (c *Client) CreateCollection(ctx context.Context, name string, vectorSize int) error {
	body := map[string]any{
		"vectors": map[string]any{"size": vectorSize, "distance": "Cosine"},
	}
	return c.do(ctx, http.MethodPut, "/collections/"+url.PathEscape(name), body, nil)
}

// EnsureCollection creates the collection only if it does not exist yet.
func (c *Client) EnsureCollection(ctx context.Context, name string, vectorSize int) error {
	exists, err := c.CollectionExists(ctx, name)
	if err != nil {
		return err
	}
	if exists {
		return nil
	}
	return c.CreateCollection(ctx, name, vectorSize)
}

// DeleteCollection drops a collection and all of its points.
func (c *Client) DeleteCollection(ctx context.Context, name string) error {
	return c.do(ctx, http.MethodDelete, "/collections/"+url.PathEscape(name), nil, nil)
}

// Upsert inserts or replaces points and waits until they are indexed.
func (c *Client) Upsert(ctx context.Context, collection string, points []Point) error {
	if len(points) == 0 {
		return nil
	}
	body := map[string]any{"points": points}
	return c.do(ctx, http.MethodPut, "/collections/"+url.PathEscape(collection)+"/points?wait=true", body, nil)
}

// Search returns the top `limit` points closest to vector. filter may be nil.
func (c *Client) Search(ctx context.Context, collection string, vector []float32, limit int, filter *Filter) ([]ScoredPoint, error) {
	body := map[string]any{
		"query":        vector,
		"limit":        limit,
		"with_payload": true,
	}
	if filter != nil {
		body["filter"] = filter
	}
	var res struct {
		Points []ScoredPoint `json:"points"`
	}
	err := c.do(ctx, http.MethodPost, "/collections/"+url.PathEscape(collection)+"/points/query", body, &res)
	return res.Points, err
}

type Record struct {
	ID      any            `json:"id"`
	Payload map[string]any `json:"payload"`
}

// Scroll pages through points matching filter without vectors. Pass the
// returned next offset back in to get the following page; nil means done.
func (c *Client) Scroll(ctx context.Context, collection string, filter *Filter, limit int, offset any) ([]Record, any, error) {
	body := map[string]any{
		"limit":        limit,
		"with_payload": true,
		"with_vector":  false,
	}
	if filter != nil {
		body["filter"] = filter
	}
	if offset != nil {
		body["offset"] = offset
	}
	var res struct {
		Points         []Record `json:"points"`
		NextPageOffset any      `json:"next_page_offset"`
	}
	err := c.do(ctx, http.MethodPost, "/collections/"+url.PathEscape(collection)+"/points/scroll", body, &res)
	return res.Points, res.NextPageOffset, err
}

// CreatePayloadIndex indexes a payload field so filters on it stay fast.
// schema is a Qdrant field type such as "keyword" or "integer". Creating an
// index that already exists is a no-op.
func (c *Client) CreatePayloadIndex(ctx context.Context, collection, field, schema string) error {
	body := map[string]any{"field_name": field, "field_schema": schema}
	return c.do(ctx, http.MethodPut, "/collections/"+url.PathEscape(collection)+"/index?wait=true", body, nil)
}

// SetPayload merges payload into every point matching filter.
func (c *Client) SetPayload(ctx context.Context, collection string, payload map[string]any, filter *Filter) error {
	body := map[string]any{"payload": payload, "filter": filter}
	return c.do(ctx, http.MethodPost, "/collections/"+url.PathEscape(collection)+"/points/payload?wait=true", body, nil)
}

// DeleteByFilter removes every point matching filter.
func (c *Client) DeleteByFilter(ctx context.Context, collection string, filter *Filter) error {
	body := map[string]any{"filter": filter}
	return c.do(ctx, http.MethodPost, "/collections/"+url.PathEscape(collection)+"/points/delete?wait=true", body, nil)
}

// do sends a request and decodes the "result" field of Qdrant's response into out.
func (c *Client) do(ctx context.Context, method, path string, in, out any) error {
	var reqBody io.Reader
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return err
		}
		reqBody = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reqBody)
	if err != nil {
		return err
	}
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set("api-key", c.apiKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("qdrant %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode >= 300 {
		return fmt.Errorf("qdrant %s %s: status %d: %s", method, path, resp.StatusCode, bytes.TrimSpace(raw))
	}
	if out == nil {
		return nil
	}

	var envelope struct {
		Result json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return fmt.Errorf("qdrant %s %s: decode: %w", method, path, err)
	}
	return json.Unmarshal(envelope.Result, out)
}

// NewID returns a random UUIDv4 string, suitable as a Qdrant point ID.
func NewID() string {
	var b [16]byte
	rand.Read(b[:])
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
