// Package pdftext extracts plain text from PDF files, one entry per page.
package pdftext

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/ledongthuc/pdf"
)

type Page struct {
	Number int // 1-based
	Text   string
}

// Extract returns the text of every non-empty page. Scanned PDFs without a
// text layer yield no pages; OCR is out of scope.
func Extract(data []byte) (pages []Page, err error) {
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
			continue
		}
		text, err := p.GetPlainText(nil)
		if err != nil {
			return nil, fmt.Errorf("pdf: page %d: %w", i, err)
		}
		if text = strings.TrimSpace(text); text != "" {
			pages = append(pages, Page{Number: i, Text: text})
		}
	}
	return pages, nil
}
