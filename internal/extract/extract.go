// Package extract turns uploaded files into text sections for indexing.
// The file type is detected from its content, not its name, except for plain
// text where only the extension can tell Markdown from TXT.
package extract

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/nabildzr/rag-chatbot/internal/pdftext"
)

type Kind string

const (
	PDF   Kind = "pdf"
	DOCX  Kind = "docx"
	PPTX  Kind = "pptx"
	TXT   Kind = "txt"
	MD    Kind = "md"
	Image Kind = "image" // PNG or JPEG, read with OCR
)

// Unit names what a section number refers to, for citations.
func (k Kind) Unit() string {
	switch k {
	case PDF:
		return "page"
	case PPTX:
		return "slide"
	case Image:
		return "image"
	default:
		return "section"
	}
}

var ErrUnsupported = errors.New("format tidak didukung")

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

// OCR reads text from images; see package ocr.
type OCR interface {
	Image(ctx context.Context, data []byte) (string, error)
	PDFPage(ctx context.Context, pdf []byte, page int) (string, error)
	CanReadPDF() bool
}

// Extractor turns uploads into sections. With OCR set it also accepts
// images and reads PDF pages that have no text layer.
type Extractor struct {
	OCR         OCR
	MaxOCRPages int // per PDF; 0 means 50
}

// Extensions lists the file types the upload form accepts.
func (e Extractor) Extensions() []string {
	ext := []string{".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown"}
	if e.OCR != nil {
		ext = append(ext, ".png", ".jpg", ".jpeg")
	}
	return ext
}

func (e Extractor) unsupported() error {
	names := "PDF, DOCX, PPTX, TXT, MD"
	if e.OCR != nil {
		names += ", PNG, JPG"
	}
	return fmt.Errorf("%w (%s)", ErrUnsupported, names)
}

// Extract detects the format of data and returns its non-empty sections.
func (e Extractor) Extract(ctx context.Context, filename string, data []byte) (*Document, error) {
	kind, err := e.Detect(filename, data)
	if err != nil {
		return nil, err
	}
	var sections []Section
	switch kind {
	case PDF:
		sections, err = e.extractPDF(ctx, data)
	case Image:
		var text string
		text, err = e.OCR.Image(ctx, data)
		if text = strings.TrimSpace(text); text != "" {
			sections = []Section{{Number: 1, Text: text}}
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

// extractPDF uses the text layer, and OCR for pages that have none (scans)
// when OCR is available.
func (e Extractor) extractPDF(ctx context.Context, data []byte) ([]Section, error) {
	pages, err := pdftext.ExtractAll(data)
	if err != nil {
		return nil, err
	}
	limit := e.MaxOCRPages
	if limit == 0 {
		limit = 50
	}
	canOCR := e.OCR != nil && e.OCR.CanReadPDF()
	var sections []Section
	ocred := 0
	for _, p := range pages {
		text := p.Text
		// A handful of characters (page numbers, stray marks) is not a text layer.
		if canOCR && len(strings.Fields(text)) < 5 && ocred < limit {
			ocred++
			t, err := e.OCR.PDFPage(ctx, data, p.Number)
			if err != nil {
				return nil, fmt.Errorf("ocr page %d: %w", p.Number, err)
			}
			text = t
		}
		if text = strings.TrimSpace(text); text != "" {
			sections = append(sections, Section{Number: p.Number, Text: text})
		}
	}
	return sections, nil
}

// Detect identifies the format from magic bytes, falling back to the
// extension for plain text.
func (e Extractor) Detect(filename string, data []byte) (Kind, error) {
	switch {
	case bytes.HasPrefix(data, []byte("%PDF-")):
		return PDF, nil
	case bytes.HasPrefix(data, []byte{'P', 'K', 3, 4}):
		return detectOOXML(data)
	case bytes.HasPrefix(data, []byte{0x89, 'P', 'N', 'G'}), bytes.HasPrefix(data, []byte{0xff, 0xd8, 0xff}):
		if e.OCR == nil {
			return "", fmt.Errorf("gambar butuh OCR, yang belum aktif di server: %w", ErrUnsupported)
		}
		return Image, nil
	}
	ext := strings.ToLower(filepath.Ext(filename))
	if ext != ".txt" && ext != ".md" && ext != ".markdown" {
		return "", e.unsupported()
	}
	if !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 {
		return "", fmt.Errorf("file teks harus UTF-8: %w", ErrUnsupported)
	}
	if ext == ".txt" {
		return TXT, nil
	}
	return MD, nil
}

// Extract and Detect without OCR, for callers that do not need it.
func Extract(filename string, data []byte) (*Document, error) {
	return Extractor{}.Extract(context.Background(), filename, data)
}

func Detect(filename string, data []byte) (Kind, error) { return Extractor{}.Detect(filename, data) }

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
