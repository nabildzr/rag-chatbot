// Package pdftext extracts plain text from PDF files, one entry per page.
package pdftext

import (
	"bytes"
	"fmt"
	"math"
	"strings"
	"unicode"

	"github.com/ledongthuc/pdf"
)

type Page struct {
	Number int // 1-based
	Text   string
}

// Extract returns the text of every non-empty page. Scanned pages without a
// text layer are left out; see ExtractAll.
func Extract(data []byte) ([]Page, error) {
	all, err := ExtractAll(data)
	if err != nil {
		return nil, err
	}
	pages := all[:0]
	for _, p := range all {
		if p.Text != "" {
			pages = append(pages, p)
		}
	}
	return pages, nil
}

// ExtractAll returns every page, with empty Text for pages that have no text
// layer (scans), so callers can OCR them.
func ExtractAll(data []byte) (pages []Page, err error) {
	// The parser panics on some malformed files; turn that into an error.
	defer func() {
		if r := recover(); r != nil {
			pages, err = nil, fmt.Errorf("pdf: malformed file: %v", r)
		}
	}()

	r, err := pdf.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, fmt.Errorf("pdf: %w", err)
	}

	for i := 1; i <= r.NumPage(); i++ {
		p := r.Page(i)
		if p.V.IsNull() {
			pages = append(pages, Page{Number: i})
			continue
		}
		pages = append(pages, Page{Number: i, Text: assemble(p.Content().Text)})
	}
	return pages, nil
}

// Glyph gaps wider than this fraction of the font size become spaces. Many
// PDFs (LaTeX output in particular) never draw space characters and only
// move the pen, so the library's GetPlainText glues words together.
const spaceGap = 0.15

// assemble rebuilds text from positioned glyphs in content-stream order:
// a horizontal gap inserts a space, a change of baseline starts a new line,
// and a word hyphenated across lines is joined back together.
func assemble(glyphs []pdf.Text) string {
	var b []byte
	sep := func(c byte) { // add a separator unless one is already there
		if n := len(b); n > 0 && b[n-1] != '\n' && (c == '\n' || b[n-1] != ' ') {
			if c == '\n' && b[n-1] == ' ' {
				b[n-1] = '\n'
				return
			}
			b = append(b, c)
		}
	}
	var prev *pdf.Text
	for i := range glyphs {
		g := &glyphs[i]
		if g.S == "" || g.S == "\n" { // the library emits "\n" after every TJ; not a real break
			continue
		}
		if prev != nil {
			size := math.Max(math.Max(prev.FontSize, g.FontSize), 1)
			sameLine := math.Abs(g.Y-prev.Y) < size*0.5
			gap := g.X - (prev.X + prev.W)
			switch {
			case !sameLine || gap < -size: // new baseline, or jumped back left (next column)
				sep('\n')
			case gap > size*spaceGap:
				sep(' ')
			}
		}
		b = append(b, g.S...)
		prev = g
	}
	return tidy(string(b))
}

// tidy joins words hyphenated across lines ("approxi-\nmating"), then
// collapses each line's whitespace.
func tidy(s string) string {
	lines := strings.Split(s, "\n")
	var out []string
	for _, line := range lines {
		line = strings.Join(strings.Fields(line), " ")
		if line == "" {
			continue
		}
		if n := len(out); n > 0 && strings.HasSuffix(out[n-1], "-") && startsLower(line) {
			out[n-1] = strings.TrimSuffix(out[n-1], "-") + line
			continue
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

func startsLower(s string) bool {
	for _, r := range s {
		return unicode.IsLower(r)
	}
	return false
}
