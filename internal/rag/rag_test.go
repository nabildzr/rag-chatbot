package rag

import (
	"strings"
	"testing"
)

func words(n int) string {
	w := make([]string, n)
	for i := range w {
		w[i] = "w"
	}
	return strings.Join(w, " ")
}

func TestChunkPagesOverlap(t *testing.T) {
	chunks := ChunkPages([]Page{{Number: 1, Text: words(25)}}, 10, 2)
	// windows start at 0, 8, 16 -> lengths 10, 10, 9
	if len(chunks) != 3 {
		t.Fatalf("got %d chunks, want 3", len(chunks))
	}
	if n := len(strings.Fields(chunks[2].Text)); n != 9 {
		t.Fatalf("last chunk has %d words, want 9", n)
	}
}

func TestChunkPagesKeepsPageAndIndex(t *testing.T) {
	chunks := ChunkPages([]Page{
		{Number: 2, Text: "alpha beta"},
		{Number: 5, Text: "gamma"},
	}, 10, 2)
	if len(chunks) != 2 {
		t.Fatalf("got %d chunks, want 2", len(chunks))
	}
	if chunks[0].Page != 2 || chunks[1].Page != 5 || chunks[1].Index != 1 || chunks[1].Text != "gamma" {
		t.Fatalf("bad chunks: %+v", chunks)
	}
}

func TestChunkPagesSkipsEmptyPages(t *testing.T) {
	if chunks := ChunkPages([]Page{{Number: 1, Text: "  \n "}}, 10, 2); len(chunks) != 0 {
		t.Fatalf("want no chunks, got %+v", chunks)
	}
}
