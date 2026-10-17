package rag

import "sort"

// Candidate is a chunk found by dense and/or keyword (BM25) search.
type Candidate struct {
	Source           // Score holds the dense cosine similarity
	DenseRank   int  // 1-based rank in the dense results; 0 if not found there
	KeywordRank int  // 1-based rank in the keyword results; 0 if not found there
	Exact       bool // the chunk contains every keyword of the question
}

// rrfK dampens the weight of top ranks in reciprocal rank fusion; 60 is the
// value from the original RRF paper and the usual default.
const rrfK = 60

// Fuse orders candidates by reciprocal rank fusion of their dense and keyword
// ranks and returns at most k of them.
//
// Filtering still uses cosine similarity, as in a dense-only search: a chunk
// is kept when it reaches minScore and lies within margin of the best cosine.
// A chunk that contains every keyword of the question is kept regardless:
// that is the case keyword search exists for (names, codes, numbers the
// embedding scores poorly). minScore and margin of 0 disable filtering.
func Fuse(cands []Candidate, minScore, margin float32, k int) []Source {
	var best float32
	for _, c := range cands {
		best = max(best, c.Score)
	}
	floor := minScore
	if margin > 0 {
		floor = max(floor, best-margin)
	}

	type scored struct {
		c   Candidate
		rrf float64
	}
	var kept []scored
	for _, c := range cands {
		if c.Score < floor && !c.Exact {
			continue
		}
		var rrf float64
		if c.DenseRank > 0 {
			rrf += 1 / float64(rrfK+c.DenseRank)
		}
		if c.KeywordRank > 0 {
			rrf += 1 / float64(rrfK+c.KeywordRank)
		}
		kept = append(kept, scored{c, rrf})
	}
	sort.SliceStable(kept, func(i, j int) bool {
		if kept[i].rrf != kept[j].rrf {
			return kept[i].rrf > kept[j].rrf
		}
		return kept[i].c.Score > kept[j].c.Score
	})

	out := []Source{}
	for _, s := range kept {
		if len(out) == k {
			break
		}
		out = append(out, s.c.Source)
	}
	return out
}
