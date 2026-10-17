package pdftext

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

// buildPDF writes a minimal valid PDF with one text line per page.
func buildPDF(lines ...string) []byte {
	streams := make([]string, len(lines))
	for i, line := range lines {
		streams[i] = fmt.Sprintf("BT /F1 12 Tf 72 720 Td (%s) Tj ET", line)
	}
	return buildPDFStreams(streams...)
}

// buildPDFStreams writes a minimal valid PDF with one raw content stream per page.
func buildPDFStreams(lines ...string) []byte {
	var objs []string
	kids := make([]string, len(lines))
	for i := range lines {
		kids[i] = fmt.Sprintf("%d 0 R", 4+2*i)
	}
	objs = append(objs,
		"<< /Type /Catalog /Pages 2 0 R >>",
		fmt.Sprintf("<< /Type /Pages /Kids [%s] /Count %d >>", strings.Join(kids, " "), len(lines)),
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
	)
	for i, line := range lines {
		stream := line
		objs = append(objs,
			fmt.Sprintf("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>", 5+2*i),
			fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(stream), stream),
		)
	}

	var b bytes.Buffer
	b.WriteString("%PDF-1.4\n")
	offsets := make([]int, len(objs))
	for i, o := range objs {
		offsets[i] = b.Len()
		fmt.Fprintf(&b, "%d 0 obj\n%s\nendobj\n", i+1, o)
	}
	xref := b.Len()
	fmt.Fprintf(&b, "xref\n0 %d\n0000000000 65535 f \n", len(objs)+1)
	for _, off := range offsets {
		fmt.Fprintf(&b, "%010d 00000 n \n", off)
	}
	fmt.Fprintf(&b, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objs)+1, xref)
	return b.Bytes()
}

func TestExtractPages(t *testing.T) {
	pages, err := Extract(buildPDF("Halo dunia", "RAG chatbot"))
	if err != nil {
		t.Fatal(err)
	}
	if len(pages) != 2 || pages[1].Number != 2 {
		t.Fatalf("bad pages: %+v", pages)
	}
	if !strings.Contains(pages[0].Text, "Halo dunia") || !strings.Contains(pages[1].Text, "RAG chatbot") {
		t.Fatalf("bad text: %+v", pages)
	}
}

func TestExtractMalformed(t *testing.T) {
	if _, err := Extract([]byte("%PDF-1.4 garbage")); err == nil {
		t.Fatal("expected error")
	}
}

// LaTeX-style output: words are separate strings positioned with TJ kerning
// instead of space characters, and a word is hyphenated across two lines.
func TestExtractPositionedWords(t *testing.T) {
	stream := "BT /F1 12 Tf 72 720 Td [(Dropout) -400 (prevents) -400 (overfit-)] TJ " +
		"0 -14 Td [(ting) -400 (in) -400 (nets.)] TJ " +
		"0 -14 Td [(Second) -400 (line.)] TJ ET"
	pages, err := Extract(buildPDFStreams(stream))
	if err != nil {
		t.Fatal(err)
	}
	want := "Dropout prevents overfitting in nets.\nSecond line."
	if len(pages) != 1 || pages[0].Text != want {
		t.Fatalf("got %+v, want %q", pages, want)
	}
}

func TestTidy(t *testing.T) {
	got := tidy("  a   b \n\n well-\nknown  \nX-\nRay")
	if got != "a b\nwellknown\nX-\nRay" {
		t.Fatalf("got %q", got)
	}
}
