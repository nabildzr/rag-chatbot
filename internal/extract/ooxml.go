package extract

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// maxPartBytes caps how much of one ZIP entry is read, so a small upload
// cannot expand into gigabytes (zip bomb).
const maxPartBytes = 50 << 20

func detectOOXML(data []byte) (Kind, error) {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return "", fmt.Errorf("arsip ZIP rusak: %w", ErrUnsupported)
	}
	for _, f := range zr.File {
		switch f.Name {
		case "word/document.xml":
			return DOCX, nil
		case "ppt/presentation.xml":
			return PPTX, nil
		}
	}
	return "", ErrUnsupported
}

func readPart(zr *zip.Reader, name string) ([]byte, error) {
	f, err := zr.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, maxPartBytes+1))
	if err != nil {
		return nil, err
	}
	if len(b) > maxPartBytes {
		return nil, fmt.Errorf("%s terlalu besar", name)
	}
	return b, nil
}

// extractDOCX reads word/document.xml paragraph by paragraph and starts a
// new section at every Title/Heading-styled paragraph. Word has no stable
// page numbers in the file, so sections stand in for them.
func extractDOCX(data []byte) ([]Section, error) {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, err
	}
	doc, err := readPart(zr, "word/document.xml")
	if err != nil {
		return nil, err
	}

	type para struct {
		text    string
		heading bool
	}
	var paras []para
	var cur strings.Builder
	heading, inText := false, false

	dec := xml.NewDecoder(bytes.NewReader(doc))
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("docx: %w", err)
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "p":
				cur.Reset()
				heading = false
			case "t":
				inText = true
			case "tab":
				cur.WriteByte('\t')
			case "br", "cr":
				cur.WriteByte('\n')
			case "pStyle":
				for _, a := range t.Attr {
					if a.Name.Local == "val" && (strings.HasPrefix(a.Value, "Heading") || a.Value == "Title") {
						heading = true
					}
				}
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "t":
				inText = false
			case "p":
				if text := strings.TrimSpace(cur.String()); text != "" {
					paras = append(paras, para{text, heading})
				}
			}
		case xml.CharData:
			if inText {
				cur.Write(t)
			}
		}
	}

	var sections []Section
	var title string
	var body []string
	words := 0
	flush := func() {
		if len(body) > 0 {
			sections = append(sections, Section{Number: len(sections) + 1, Title: title, Text: strings.Join(body, "\n")})
			body, words = nil, 0
		}
	}
	for _, p := range paras {
		if p.heading {
			flush()
			title = p.text
		}
		body = append(body, p.text)
		words += len(strings.Fields(p.text))
		// Long stretches without headings still get split for citations.
		if words > 2*sectionWords {
			flush()
		}
	}
	flush()
	return sections, nil
}

var slideFile = regexp.MustCompile(`^ppt/slides/slide(\d+)\.xml$`)

// extractPPTX returns one section per slide, in presentation order.
func extractPPTX(data []byte) ([]Section, error) {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, err
	}
	var sections []Section
	for i, name := range slideOrder(zr) {
		part, err := readPart(zr, name)
		if err != nil {
			return nil, err
		}
		text, err := drawingText(part)
		if err != nil {
			return nil, fmt.Errorf("pptx %s: %w", name, err)
		}
		if text != "" {
			sections = append(sections, Section{Number: i + 1, Text: text})
		}
	}
	return sections, nil
}

// slideOrder follows presentation.xml's slide list through its relationships;
// slide file numbers do not change when slides are reordered. If that fails
// it falls back to file number order.
func slideOrder(zr *zip.Reader) []string {
	var pres struct {
		Slides []struct {
			RID string `xml:"http://schemas.openxmlformats.org/officeDocument/2006/relationships id,attr"`
		} `xml:"sldIdLst>sldId"`
	}
	var rels struct {
		Rels []struct {
			ID     string `xml:"Id,attr"`
			Target string `xml:"Target,attr"`
		} `xml:"Relationship"`
	}
	presXML, err1 := readPart(zr, "ppt/presentation.xml")
	relsXML, err2 := readPart(zr, "ppt/_rels/presentation.xml.rels")
	if err1 == nil && err2 == nil && xml.Unmarshal(presXML, &pres) == nil && xml.Unmarshal(relsXML, &rels) == nil {
		targets := map[string]string{}
		for _, r := range rels.Rels {
			targets[r.ID] = path.Join("ppt", r.Target)
		}
		var order []string
		for _, s := range pres.Slides {
			if t, ok := targets[s.RID]; ok && slideFile.MatchString(t) {
				order = append(order, t)
			}
		}
		if len(order) > 0 {
			return order
		}
	}

	type numbered struct {
		n    int
		name string
	}
	var files []numbered
	for _, f := range zr.File {
		if m := slideFile.FindStringSubmatch(f.Name); m != nil {
			n, _ := strconv.Atoi(m[1])
			files = append(files, numbered{n, f.Name})
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].n < files[j].n })
	order := make([]string, len(files))
	for i, f := range files {
		order[i] = f.name
	}
	return order
}

// drawingText collects DrawingML text runs (<a:t>), one line per paragraph.
func drawingText(part []byte) (string, error) {
	var lines []string
	var cur strings.Builder
	inText := false
	dec := xml.NewDecoder(bytes.NewReader(part))
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		switch t := tok.(type) {
		case xml.StartElement:
			if t.Name.Local == "t" {
				inText = true
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "t":
				inText = false
			case "p":
				if line := strings.TrimSpace(cur.String()); line != "" {
					lines = append(lines, line)
				}
				cur.Reset()
			}
		case xml.CharData:
			if inText {
				cur.Write(t)
			}
		}
	}
	return strings.Join(lines, "\n"), nil
}
