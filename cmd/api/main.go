package main

import (
	"errors"
	"io/fs"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/nabildzr/rag-chatbot/internal/embed"
	"github.com/nabildzr/rag-chatbot/internal/handler"
	"github.com/nabildzr/rag-chatbot/internal/llm"
	"github.com/nabildzr/rag-chatbot/internal/pdftext"
	"github.com/nabildzr/rag-chatbot/internal/rag"
	"github.com/nabildzr/rag-chatbot/internal/vector"
)

func main() {
	if err := loadDotEnv(".env"); err != nil {
		log.Fatal(err)
	}
	port := getenv("PORT", "8080")

	groqKey := os.Getenv("GROQ_API_KEY")
	if groqKey == "" {
		log.Print("warning: GROQ_API_KEY is not set; /chat will fail until it is")
	}

	h := handler.New(handler.Deps{
		Embedder:   embed.New(getenv("OLLAMA_URL", "http://localhost:11434"), getenv("EMBED_MODEL", "nomic-embed-text")),
		Vectors:    vector.New(getenv("QDRANT_URL", "http://localhost:6333"), os.Getenv("QDRANT_API_KEY")),
		LLM:        llm.New(getenv("LLM_BASE_URL", llm.DefaultBaseURL), groqKey, getenv("GROQ_MODEL", "openai/gpt-oss-120b")),
		ExtractPDF: extractPDF,
		VectorSize: embed.Dimension,

		MinScore:    getenvFloat("MIN_SCORE", rag.DefaultMinScore),
		ScoreMargin: getenvFloat("SCORE_MARGIN", rag.DefaultScoreMargin),
	})

	log.Printf("API listening on :%s", port)
	if err := http.ListenAndServe(":"+port, h.Routes()); err != nil {
		log.Fatal(err)
	}
}

func extractPDF(data []byte) ([]rag.Page, error) {
	pages, err := pdftext.Extract(data)
	if err != nil {
		return nil, err
	}
	out := make([]rag.Page, len(pages))
	for i, p := range pages {
		out[i] = rag.Page{Number: p.Number, Text: p.Text}
	}
	return out, nil
}

// loadDotEnv sets KEY=VALUE lines from path for local development. Variables
// already set in the environment win, so production config is never overridden.
// A missing file is not an error.
func loadDotEnv(path string) error {
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		if _, set := os.LookupEnv(key); !set {
			os.Setenv(key, strings.Trim(strings.TrimSpace(value), `"'`))
		}
	}
	return nil
}

func getenvFloat(key string, fallback float32) float32 {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	f, err := strconv.ParseFloat(v, 32)
	if err != nil {
		log.Fatalf("%s: invalid number %q", key, v)
	}
	return float32(f)
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
