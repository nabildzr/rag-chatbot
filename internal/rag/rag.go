// Package rag handles chunking and prompt building.
package rag

import "strings"

// Chunk sizes are counted in words. One token is roughly 0.75 English words,
// so 375/40 words approximates the planned 500/50 tokens.
const (
	DefaultChunkWords   = 375
	DefaultOverlapWords = 40
)

type Page struct {
	Number int
	Text   string
}

type Chunk struct {
	Index int // position within the document, 0-based
	Page  int // page the chunk starts on
	Text  string
}

// ChunkPages splits pages into overlapping word windows. Chunks never cross
// a page boundary, so every chunk cites exactly one page.
func ChunkPages(pages []Page, size, overlap int) []Chunk {
	if size <= 0 {
		size = DefaultChunkWords
	}
	if overlap < 0 || overlap >= size {
		overlap = 0
	}
	step := size - overlap

	var chunks []Chunk
	for _, p := range pages {
		words := strings.Fields(p.Text)
		for start := 0; start < len(words); start += step {
			end := min(start+size, len(words))
			chunks = append(chunks, Chunk{
				Index: len(chunks),
				Page:  p.Number,
				Text:  strings.Join(words[start:end], " "),
			})
			if end == len(words) {
				break
			}
		}
	}
	return chunks
}
