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

func TestSelectSources(t *testing.T) {
	src := func(scores ...float32) []Source {
		out := make([]Source, len(scores))
		for i, s := range scores {
			out[i] = Source{Score: s, ChunkIndex: i}
		}
		return out
	}
	cases := []struct {
		name          string
		in            []Source
		min, margin   float32
		wantKeptCount int
	}{
		{"relative margin drops weak tail", src(0.77, 0.61, 0.60), 0.64, 0.10, 1},
		{"close scores all kept", src(0.81, 0.75, 0.72), 0.64, 0.10, 3},
		{"off-topic question keeps nothing", src(0.63, 0.61, 0.60), 0.64, 0.10, 0},
		{"zero margin uses only floor", src(0.90, 0.65, 0.60), 0.64, 0, 2},
		{"zero floor and margin keeps all", src(0.30, 0.10), 0, 0, 2},
		{"empty input", nil, 0.64, 0.10, 0},
	}
	for _, c := range cases {
		if got := SelectSources(c.in, c.min, c.margin); len(got) != c.wantKeptCount {
			t.Errorf("%s: kept %d, want %d", c.name, len(got), c.wantKeptCount)
		}
	}
}

func TestStripCitations(t *testing.T) {
	if got := StripCitations("14 hari [1][2] dan Rp75.000 【3】."); got != "14 hari dan Rp75.000." {
		t.Fatalf("got %q", got)
	}
}

func TestCitedNumbers(t *testing.T) {
	got := CitedNumbers("Cuti 14 hari [2]. Lembur Rp75.000【1】, lihat juga [2][3] dan [x].")
	want := []int{2, 1, 3}
	if len(got) != len(want) || got[0] != 2 || got[1] != 1 || got[2] != 3 {
		t.Fatalf("got %v, want %v", got, want)
	}
}
