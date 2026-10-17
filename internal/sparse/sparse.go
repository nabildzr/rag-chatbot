// Package sparse builds BM25-style sparse vectors for keyword search in
// Qdrant. Term frequency saturation is computed here; the IDF part is left
// to Qdrant (sparse vector modifier "idf"), which knows the whole collection.
package sparse

import (
	"hash/fnv"
	"sort"
	"strings"
	"unicode"
)

// Vector is a Qdrant sparse vector.
type Vector struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

// BM25 parameters; avgLen approximates the token count of one chunk.
const (
	k1     = 1.2
	b      = 0.75
	avgLen = 300
)

// Document encodes chunk text with saturated term frequencies.
func Document(text string) Vector {
	tokens := Tokenize(text)
	tf := map[uint32]float64{}
	for _, t := range tokens {
		tf[hash(t)]++
	}
	norm := k1 * (1 - b + b*float64(len(tokens))/avgLen)
	return build(tf, func(f float64) float64 { return f * (k1 + 1) / (f + norm) })
}

// Query encodes a question: every distinct term counts once.
func Query(text string) Vector {
	tf := map[uint32]float64{}
	for _, t := range Tokenize(text) {
		tf[hash(t)] = 1
	}
	return build(tf, func(f float64) float64 { return f })
}

func build(tf map[uint32]float64, weight func(float64) float64) Vector {
	v := Vector{Indices: make([]uint32, 0, len(tf)), Values: make([]float32, 0, len(tf))}
	for idx := range tf {
		v.Indices = append(v.Indices, idx)
	}
	sort.Slice(v.Indices, func(i, j int) bool { return v.Indices[i] < v.Indices[j] })
	for _, idx := range v.Indices {
		v.Values = append(v.Values, float32(weight(tf[idx])))
	}
	return v
}

// Tokenize lowercases text and splits it into words and numbers. Separators
// inside numbers and identifiers ("3.500", "rag-chatbot", "/readyz") split
// too, so "readyz" and "3" "500" match however they were written. Stop words
// and single letters are dropped.
func Tokenize(text string) []string {
	fields := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	out := fields[:0]
	for _, f := range fields {
		if len([]rune(f)) < 2 && !unicode.IsDigit([]rune(f)[0]) {
			continue
		}
		if stopwords[f] {
			continue
		}
		out = append(out, f)
	}
	return out
}

// hash maps a term to a sparse index. Collisions are possible but rare
// enough at 32 bits to not matter for ranking.
func hash(term string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(term))
	return h.Sum32()
}

// Common English and Indonesian function words.
var stopwords = func() map[string]bool {
	m := map[string]bool{}
	for _, w := range strings.Fields(`
		the a an and or of to in on at for with by from is are was were be been
		it this that these those as do does did how what which who whom when
		where why can could should would will i you he she we they my your our
		yang dan di ke dari untuk dengan pada ini itu adalah atau juga dalam
		tidak akan bisa ada saya kamu kami kita mereka apa siapa kapan mana
		bagaimana berapa kenapa mengapa sebagai oleh karena jika kalau sudah
		harus boleh para setiap tersebut`) {
		m[w] = true
	}
	return m
}()

// ContainsAll reports whether text contains every keyword of query (after
// tokenizing both). A query without keywords contains nothing.
func ContainsAll(text, query string) bool {
	want := Tokenize(query)
	if len(want) == 0 {
		return false
	}
	have := map[string]bool{}
	for _, t := range Tokenize(text) {
		have[t] = true
	}
	for _, t := range want {
		if !have[t] {
			return false
		}
	}
	return true
}
