package rag

import (
	"fmt"
	"regexp"
	"strings"
)

// Source is a retrieved chunk shown to the model and returned as a citation.
type Source struct {
	DocumentID string  `json:"document_id"`
	Filename   string  `json:"filename"`
	Page       int     `json:"page"`
	ChunkIndex int     `json:"chunk_index"`
	Score      float32 `json:"score"`
	Text       string  `json:"text"`
}

// Defaults tuned on nomic-embed-text with cosine distance: relevant chunks
// scored 0.65-0.81, the best chunk for off-topic questions 0.60-0.63.
const (
	DefaultMinScore    = 0.64
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

// BuildUserPrompt numbers each source as [1]..[n] and appends the question.
func BuildUserPrompt(question string, sources []Source) string {
	var b strings.Builder
	b.WriteString("Context:\n\n")
	for i, s := range sources {
		fmt.Fprintf(&b, "[%d] (%s, page %d)\n%s\n\n", i+1, s.Filename, s.Page, s.Text)
	}
	b.WriteString("Question: ")
	b.WriteString(question)
	return b.String()
}
