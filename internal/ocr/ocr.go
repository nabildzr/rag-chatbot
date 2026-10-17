// Package ocr reads text from images and scanned PDF pages with the
// Tesseract and Poppler (pdftoppm) command-line tools.
package ocr

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// pageTimeout bounds one OCR or render call; a scan page takes a few seconds.
const pageTimeout = 2 * time.Minute

type Tesseract struct {
	langs     string // e.g. "ind+eng"
	tesseract string
	pdftoppm  string // empty: scanned PDFs are not supported
}

// Find locates the tools on PATH. It returns an error when tesseract is
// missing; pdftoppm is optional and only needed for scanned PDFs.
func Find(langs string) (*Tesseract, error) {
	t, err := exec.LookPath("tesseract")
	if err != nil {
		return nil, errors.New("tesseract not found on PATH")
	}
	p, _ := exec.LookPath("pdftoppm")
	return &Tesseract{langs: langs, tesseract: t, pdftoppm: p}, nil
}

// CanReadPDF reports whether scanned PDF pages can be rendered for OCR.
func (t *Tesseract) CanReadPDF() bool { return t.pdftoppm != "" }

// Image returns the text in an image (PNG or JPEG).
func (t *Tesseract) Image(ctx context.Context, data []byte) (string, error) {
	dir, err := os.MkdirTemp("", "ocr-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "image")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return "", err
	}
	return t.recognize(ctx, path)
}

// PDFPage renders one page (1-based) of a PDF and returns its text.
func (t *Tesseract) PDFPage(ctx context.Context, pdf []byte, page int) (string, error) {
	if t.pdftoppm == "" {
		return "", errors.New("pdftoppm not found on PATH")
	}
	dir, err := os.MkdirTemp("", "ocr-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(dir)
	in := filepath.Join(dir, "doc.pdf")
	if err := os.WriteFile(in, pdf, 0o600); err != nil {
		return "", err
	}

	ctx, cancel := context.WithTimeout(ctx, pageTimeout)
	defer cancel()
	n := strconv.Itoa(page)
	// 300 dpi grayscale is Tesseract's sweet spot for printed text.
	cmd := exec.CommandContext(ctx, t.pdftoppm, "-f", n, "-l", n, "-r", "300", "-gray", "-png", "-singlefile", in, filepath.Join(dir, "page"))
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("pdftoppm page %d: %v: %s", page, err, bytes.TrimSpace(out))
	}
	return t.recognize(ctx, filepath.Join(dir, "page.png"))
}

func (t *Tesseract) recognize(ctx context.Context, path string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, pageTimeout)
	defer cancel()
	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, t.tesseract, path, "stdout", "-l", t.langs)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("tesseract: %v: %s", err, bytes.TrimSpace(stderr.Bytes()))
	}
	return strings.TrimSpace(stdout.String()), nil
}
