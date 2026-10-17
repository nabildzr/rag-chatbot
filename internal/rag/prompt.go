package rag

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Source is a retrieved chunk shown to the model and returned as a citation.
type Source struct {
	DocumentID string  `json:"document_id"`
	Filename   string  `json:"filename"`
	Kind       string  `json:"kind"`                    // pdf, docx, pptx, txt, md
	Title      string  `json:"section_title,omitempty"` // heading of the section, if any
	Page       int     `json:"page"`                    // page, slide or section number
	ChunkIndex int     `json:"chunk_index"`
	Score      float32 `json:"score"`
	Text       string  `json:"text"`
	// N is the number the source had in the prompt, which answers cite as [N].
	N int `json:"n"`
}

// Defaults tuned with cmd/eval on bge-m3 (cosine): the expected chunk scored
// at least 0.482 (median 0.65), the best chunk for off-topic questions at
// most 0.475 (median 0.38). The floor
// sits slightly below the gap, preferring recall: a weak extra chunk only
// costs a model call that answers "not found", a missed one loses the answer.
// nomic-embed-text scores higher overall; use MIN_SCORE=0.64 with it.
const (
	DefaultMinScore    = 0.45
	DefaultScoreMargin = 0.10
)

// SelectSources keeps sources that score at least minScore and are within
// margin of the best score, so weak matches are neither sent to the model
// nor shown as citations. sources must be sorted by score, best first.
// A margin <= 0 disables the relative cut.
func SelectSources(sources []Source, minScore, margin float32) []Source {
	if len(sources) == 0 {
		return sources
	}
	cutoff := minScore
	if margin > 0 {
		cutoff = max(cutoff, sources[0].Score-margin)
	}
	kept := sources[:0:0]
	for _, s := range sources {
		if s.Score >= cutoff {
			kept = append(kept, s)
		}
	}
	return kept
}

const SystemPrompt = `You answer questions using only the numbered context passages provided.
Rules:
- Cite the passages you use with their numbers in plain ASCII square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, say you could not find it in the documents. Do not guess.
- Earlier turns of the conversation are given for context only; facts must still come from the passages.
- Answer in the same language as the question.
- Be concise.`

var citationRe = regexp.MustCompile(`\s*(?:\[\d+\]|【\d+】)`)

// StripCitations removes [n] markers. Old answers in the history refer to an
// earlier numbering of sources, which would confuse the model.
func StripCitations(s string) string {
	return citationRe.ReplaceAllString(s, "")
}

// RetrievalQuery is the text embedded for search. Follow-ups such as
// "and for overtime?" carry little meaning alone, so the previous user
// question is prepended to give the search the topic.
func RetrievalQuery(question, previousQuestion string) string {
	if previousQuestion == "" {
		return question
	}
	return previousQuestion + "\n" + question
}

// OverviewChunks is how many leading chunks of a document are used as
// context when no chunk matches the question (summary-style questions).
const OverviewChunks = 2

// BuildUserPrompt numbers each source as [1]..[n] and appends the question.
// overview marks sources that are the start of the document rather than
// search hits, so the model knows they may not answer the question.
func BuildUserPrompt(question string, sources []Source, overview bool) string {
	var b strings.Builder
	if overview {
		b.WriteString("No passage matched the question directly. Below is the beginning of the document. " +
			"Use it for questions about the document as a whole (its topic, main idea, summary). " +
			"For anything else, say you could not find it in the documents.\n\n")
	}
	b.WriteString("Context:\n\n")
	for i, s := range sources {
		fmt.Fprintf(&b, "[%d] (%s, %s %d)\n%s\n\n", i+1, s.Filename, unit(s.Kind), s.Page, s.Text)
	}
	b.WriteString("Question: ")
	b.WriteString(question)
	return b.String()
}

// unit names what Source.Page counts for a file kind.
func unit(kind string) string {
	switch kind {
	case "", "pdf":
		return "page"
	case "pptx":
		return "slide"
	case "image":
		return "image"
	default:
		return "section"
	}
}

// CitedNumbers returns the distinct source numbers an answer cites, as
// [n] or the fullwidth 【n】 some models write, in order of first use.
func CitedNumbers(answer string) []int {
	var out []int
	seen := map[int]bool{}
	for _, m := range citationNumber.FindAllStringSubmatch(answer, -1) {
		digits := m[1] + m[2]
		n, err := strconv.Atoi(digits)
		if err == nil && !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	return out
}

var citationNumber = regexp.MustCompile(`\[(\d+)\]|【(\d+)】`)
