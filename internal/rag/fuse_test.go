package rag

import "testing"

func cand(name string, score float32, denseRank, kwRank int, exact bool) Candidate {
	return Candidate{Source: Source{Filename: name, Score: score}, DenseRank: denseRank, KeywordRank: kwRank, Exact: exact}
}

func names(src []Source) []string {
	out := make([]string, len(src))
	for i, s := range src {
		out[i] = s.Filename
	}
	return out
}

func TestFuseRanksAgreementFirst(t *testing.T) {
	got := names(Fuse([]Candidate{
		cand("dense-only", 0.70, 1, 0, false),
		cand("both", 0.68, 2, 1, false),
		cand("keyword-only", 0.66, 0, 2, false),
	}, 0, 0, 5))
	want := []string{"both", "dense-only", "keyword-only"}
	if len(got) != 3 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestFuseKeepsExactMatchBelowThreshold(t *testing.T) {
	got := names(Fuse([]Candidate{
		cand("weak-semantic", 0.40, 1, 0, false),
		cand("exact-term", 0.38, 2, 1, true),
	}, 0.45, 0.10, 5))
	if len(got) != 1 || got[0] != "exact-term" {
		t.Fatalf("got %v", got)
	}
}

func TestFuseMarginAndLimit(t *testing.T) {
	got := names(Fuse([]Candidate{
		cand("a", 0.80, 1, 0, false),
		cand("b", 0.75, 2, 0, false),
		cand("c", 0.65, 3, 0, false), // more than 0.10 below the best
		cand("d", 0.78, 4, 0, false),
	}, 0.45, 0.10, 2))
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("got %v", got)
	}
}

func TestFuseEmpty(t *testing.T) {
	if got := Fuse(nil, 0.45, 0.1, 5); got == nil || len(got) != 0 {
		t.Fatalf("want empty non-nil slice, got %#v", got)
	}
}
