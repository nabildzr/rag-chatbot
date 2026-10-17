// Package extract turns uploaded files into text sections for indexing.
// The file type is detected from its content, not its name, except for plain
// text where only the extension can tell Markdown from TXT.
package extract

import (
	"bytes"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/nabildzr/rag-chatbot/internal/pdftext"
)

type Kind string

const (
	PDF  Kind = "pdf"
	DOCX Kind = "docx"
	PPTX Kind = "pptx"
	TXT  Kind = "txt"
	MD   Kind = "md"
)

// Unit names what a section number refers to, for citations.
func (k Kind) Unit() string {
	switch k {
	case PDF:
		return "page"
	case PPTX:
		return "slide"
	default:
		return "section"
	}
}

// Extensions accepted by the upload form.
var Extensions = []string{".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown"}

var ErrUnsupported = errors.New("format tidak didukung (PDF, DOCX, PPTX, TXT, MD)")

// Section is a citable unit: a PDF page, a slide, or a part of a text
// document split at headings.
type Section struct {
	Number int // 1-based
	Title  string
	Text   string
}

type Document struct {
	Kind     Kind
	Sections []Section
}

// Extract detects the format of data and returns its non-empty sections.
func Extract(filename string, data []byte) (*Document, error) {
	kind, err := Detect(filename, data)
	if err != nil {
		return nil, err
	}
	var sections []Section
	switch kind {
	case PDF:
		pages, err := pdftext.Extract(data)
		if err != nil {
			return nil, err
		}
		for _, p := range pages {
			sections = append(sections, Section{Number: p.Number, Text: p.Text})
		}
	case DOCX:
		sections, err = extractDOCX(data)
	case PPTX:
		sections, err = extractPPTX(data)
	case MD:
		sections = splitMarkdown(decodeText(data))
	case TXT:
		sections = groupParagraphs(strings.Split(decodeText(data), "\n\n"), sectionWords)
	}
	if err != nil {
		return nil, err
	}
	return &Document{Kind: kind, Sections: sections}, nil
}

// Detect identifies the format from magic bytes, falling back to the
// extension for plain text.
func Detect(filename string, data []byte) (Kind, error) {
	switch {
	case bytes.HasPrefix(data, []byte("%PDF-")):
		return PDF, nil
	case bytes.HasPrefix(data, []byte("PK\x03\x04")):
		return detectOOXML(data)
	}
	ext := strings.ToLower(filepath.Ext(filename))
	if ext != ".txt" && ext != ".md" && ext != ".markdown" {
		return "", ErrUnsupported
	}
	if !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 {
		return "", fmt.Errorf("file teks harus UTF-8: %w", ErrUnsupported)
	}
	if ext == ".txt" {
		return TXT, nil
	}
	return MD, nil
}

func decodeText(data []byte) string {
	s := strings.TrimPrefix(string(data), "FEFF")
	return strings.ReplaceAll(s, "\r\n", "\n")
}

// sectionWords is the target size of a plain-text section, matching one chunk.
const sectionWords = 375

// groupParagraphs packs paragraphs into numbered sections of about maxWords,
// so a citation points at a small, findable part of the file.
func groupParagraphs(paragraphs []string, maxWords int) []Section {
	var sections []Section
	var cur []string
	words := 0
	flush := func() {
		if len(cur) > 0 {
			sections = append(sections, Section{Number: len(sections) + 1, Text: strings.Join(cur, "\n\n")})
			cur, words = nil, 0
		}
	}
	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n := len(strings.Fields(p))
		if words > 0 && words+n > maxWords {
			flush()
		}
		cur = append(cur, p)
		words += n
	}
	flush()
	return sections
}

// splitMarkdown starts a new section at every heading (outside code fences).
// The heading line stays in the text so it is embedded with its content.
func splitMarkdown(s string) []Section {
	var sections []Section
	var title string
	var lines []string
	fenced := false
	flush := func() {
		if text := strings.TrimSpace(strings.Join(lines, "\n")); text != "" {
			sections = append(sections, Section{Number: len(sections) + 1, Title: title, Text: text})
		}
		lines = nil
	}
	for _, line := range strings.Split(s, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			fenced = !fenced
		}
		if !fenced && isHeading(trimmed) {
			flush()
			title = strings.TrimSpace(strings.TrimLeft(trimmed, "#"))
		}
		lines = append(lines, line)
	}
	flush()
	return sections
}

func isHeading(line string) bool {
	level := len(line) - len(strings.TrimLeft(line, "#"))
	return level >= 1 && level <= 6 && len(line) > level && line[level] == ' '
}
