package extract

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func zipOf(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, body := range files {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		w.Write([]byte(body))
	}
	zw.Close()
	return buf.Bytes()
}

const wns = `xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"`

func TestDOCXSplitsAtHeadings(t *testing.T) {
	doc := `<w:document ` + wns + `><w:body>
	<w:p><w:r><w:t>Kata pengantar.</w:t></w:r></w:p>
	<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Bab 1</w:t></w:r></w:p>
	<w:p><w:r><w:t xml:space="preserve">Jam kerja </w:t></w:r><w:r><w:t>08.30.</w:t></w:r><w:r><w:tab/><w:t>Catatan</w:t></w:r></w:p>
	<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Bab 2</w:t></w:r></w:p>
	<w:p><w:r><w:t>Cuti 14 hari.</w:t></w:r></w:p>
	</w:body></w:document>`
	data := zipOf(t, map[string]string{"word/document.xml": doc, "[Content_Types].xml": "<Types/>"})

	d, err := Extract("laporan.docx", data)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DOCX || len(d.Sections) != 3 {
		t.Fatalf("got %s with %+v", d.Kind, d.Sections)
	}
	if s := d.Sections[1]; s.Number != 2 || s.Title != "Bab 1" || s.Text != "Bab 1\nJam kerja 08.30.\tCatatan" {
		t.Fatalf("bad section 2: %+v", s)
	}
	if d.Sections[2].Title != "Bab 2" {
		t.Fatalf("bad section 3: %+v", d.Sections[2])
	}
}

const ans = `xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"`

func slide(lines ...string) string {
	var b strings.Builder
	b.WriteString(`<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" ` + ans + `><p:cSld><p:spTree><p:sp><p:txBody>`)
	for _, l := range lines {
		b.WriteString(`<a:p><a:r><a:t>` + l + `</a:t></a:r></a:p>`)
	}
	b.WriteString(`</p:txBody></p:sp></p:spTree></p:cSld></p:sld>`)
	return b.String()
}

func TestPPTXFollowsPresentationOrder(t *testing.T) {
	// slide2.xml is shown first: the order comes from presentation.xml.
	data := zipOf(t, map[string]string{
		"ppt/presentation.xml": `<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
			<p:sldIdLst><p:sldId id="256" r:id="rId3"/><p:sldId id="257" r:id="rId2"/></p:sldIdLst></p:presentation>`,
		"ppt/_rels/presentation.xml.rels": `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
			<Relationship Id="rId2" Target="slides/slide1.xml"/><Relationship Id="rId3" Target="slides/slide2.xml"/></Relationships>`,
		"ppt/slides/slide1.xml": slide("Penutup"),
		"ppt/slides/slide2.xml": slide("Judul", "Poin pertama"),
	})

	d, err := Extract("deck.pptx", data)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != PPTX || d.Kind.Unit() != "slide" || len(d.Sections) != 2 {
		t.Fatalf("got %s %+v", d.Kind, d.Sections)
	}
	if d.Sections[0].Text != "Judul\nPoin pertama" || d.Sections[1].Number != 2 || d.Sections[1].Text != "Penutup" {
		t.Fatalf("bad order: %+v", d.Sections)
	}
}

func TestMarkdownSplitsAtHeadingsOutsideCode(t *testing.T) {
	md := "FEFFIntro line.\r\n\r\n# Setup\r\nRun it.\n```\n# not a heading\n```\n## Usage\nAsk."
	d, err := Extract("README.md", []byte(md))
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != MD || len(d.Sections) != 3 {
		t.Fatalf("got %+v", d.Sections)
	}
	if d.Sections[1].Title != "Setup" || !strings.Contains(d.Sections[1].Text, "# not a heading") || d.Sections[2].Title != "Usage" {
		t.Fatalf("bad sections: %+v", d.Sections)
	}
}

func TestTXTGroupsParagraphs(t *testing.T) {
	para := strings.Repeat("kata ", 200)
	d, err := Extract("notes.txt", []byte(para+"\n\n"+para+"\n\n"+"pendek"))
	if err != nil {
		t.Fatal(err)
	}
	// 200 + 200 > 375, so the second paragraph starts section 2 and "pendek" joins it.
	if d.Kind != TXT || len(d.Sections) != 2 || !strings.HasSuffix(d.Sections[1].Text, "pendek") {
		t.Fatalf("got %d sections", len(d.Sections))
	}
}

func TestDetectRejectsUnknown(t *testing.T) {
	cases := map[string][]byte{
		"image.png":  {0x89, 'P', 'N', 'G'},
		"binary.txt": {'a', 0, 'b'},
		"latin1.txt": {0xff, 0xfe, 'x'},
		"sheet.xlsx": zipOf(t, map[string]string{"xl/workbook.xml": "<x/>"}),
		"notes.csv":  []byte("a,b"),
	}
	for name, data := range cases {
		if _, err := Detect(name, data); !errors.Is(err, ErrUnsupported) {
			t.Errorf("%s: got %v, want ErrUnsupported", name, err)
		}
	}
}

func TestDetectUsesContentNotName(t *testing.T) {
	if k, _ := Detect("renamed.txt", []byte("%PDF-1.7")); k != PDF {
		t.Fatalf("got %s", k)
	}
}

type fakeOCR struct{ pages []int }

func (o *fakeOCR) Image(context.Context, []byte) (string, error) { return "teks dari foto struk", nil }
func (o *fakeOCR) PDFPage(_ context.Context, _ []byte, page int) (string, error) {
	o.pages = append(o.pages, page)
	return fmt.Sprintf("hasil OCR halaman %d", page), nil
}
func (o *fakeOCR) CanReadPDF() bool { return true }

// pdfWithPages builds a PDF where an empty string means a page without text.
func pdfWithPages(texts ...string) []byte {
	var objs []string
	kids := make([]string, len(texts))
	for i := range texts {
		kids[i] = fmt.Sprintf("%d 0 R", 4+2*i)
	}
	objs = append(objs, "<< /Type /Catalog /Pages 2 0 R >>",
		fmt.Sprintf("<< /Type /Pages /Kids [%s] /Count %d >>", strings.Join(kids, " "), len(texts)),
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
	for i, t := range texts {
		stream := ""
		if t != "" {
			stream = fmt.Sprintf("BT /F1 12 Tf 72 720 Td (%s) Tj ET", t)
		}
		objs = append(objs,
			fmt.Sprintf("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>", 5+2*i),
			fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(stream), stream))
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

func TestOCRReadsScannedPagesOnly(t *testing.T) {
	ocr := &fakeOCR{}
	doc, err := Extractor{OCR: ocr}.Extract(context.Background(), "scan.pdf",
		pdfWithPages("Halaman pertama punya lapisan teks yang cukup panjang", "", "Halaman ketiga juga punya teks yang cukup panjang"))
	if err != nil {
		t.Fatal(err)
	}
	if len(ocr.pages) != 1 || ocr.pages[0] != 2 {
		t.Fatalf("OCR should run on page 2 only, ran on %v", ocr.pages)
	}
	if len(doc.Sections) != 3 || doc.Sections[1].Text != "hasil OCR halaman 2" {
		t.Fatalf("got %+v", doc.Sections)
	}
}

func TestImagesNeedOCR(t *testing.T) {
	png := []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a}
	if _, err := (Extractor{}).Detect("struk.png", png); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("without OCR: %v", err)
	}
	e := Extractor{OCR: &fakeOCR{}}
	doc, err := e.Extract(context.Background(), "struk.png", png)
	if err != nil || doc.Kind != Image || doc.Kind.Unit() != "image" || doc.Sections[0].Text != "teks dari foto struk" {
		t.Fatalf("with OCR: %+v %v", doc, err)
	}
	if !strings.Contains(strings.Join(e.Extensions(), ","), ".png") {
		t.Fatal("png should be listed with OCR")
	}
}
