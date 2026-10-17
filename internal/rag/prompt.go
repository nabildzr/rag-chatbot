package rag

import (
	"fmt"
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

const SystemPrompt = `You answer questions using only the numbered context passages provided.
Rules:
- Cite the passages you use with their numbers in plain ASCII square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, say you could not find it in the documents. Do not guess.
- Answer in the same language as the question.
- Be concise.`

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
