package main

import (
	"errors"
	"io/fs"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/nabildzr/rag-chatbot/internal/embed"
	"github.com/nabildzr/rag-chatbot/internal/extract"
	"github.com/nabildzr/rag-chatbot/internal/handler"
	"github.com/nabildzr/rag-chatbot/internal/llm"
	"github.com/nabildzr/rag-chatbot/internal/ocr"
	"github.com/nabildzr/rag-chatbot/internal/rag"
	"github.com/nabildzr/rag-chatbot/internal/ratelimit"
	"github.com/nabildzr/rag-chatbot/internal/rerank"
	"github.com/nabildzr/rag-chatbot/internal/server"
	"github.com/nabildzr/rag-chatbot/internal/store"
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
	// Only enable behind a reverse proxy that sets X-Forwarded-For itself.
	trustProxy := os.Getenv("TRUST_PROXY") == "true"

	files, err := store.New(getenv("DATA_DIR", "data"))
	if err != nil {
		log.Fatalf("DATA_DIR: %v", err)
	}

	embedModel := getenv("EMBED_MODEL", embed.DefaultModel)
	// Set below only when configured: a nil *rerank.Client must not end up
	// as a non-nil handler.Reranker interface.
	var reranker handler.Reranker
	if u := os.Getenv("RERANK_URL"); u != "" {
		reranker = rerank.New(u)
		log.Printf("reranking with %s", u)
	}
	// OCR is enabled when Tesseract is installed (the Docker image has it).
	var extractor extract.Extractor
	if os.Getenv("OCR") != "off" {
		if t, err := ocr.Find(getenv("OCR_LANGS", "ind+eng")); err == nil {
			extractor.OCR = t
			log.Printf("OCR enabled (scanned PDFs: %v)", t.CanReadPDF())
		} else {
			log.Printf("OCR disabled: %v", err)
		}
	}

	h := handler.New(handler.Deps{
		Reranker:   reranker,
		Extractor:  extractor,
		Embedder:   embed.New(getenv("OLLAMA_URL", "http://localhost:11434"), embedModel),
		Collection: getenv("COLLECTION", embed.CollectionName(embedModel)),
		DenseOnly:  os.Getenv("HYBRID_SEARCH") == "false",

		RerankMinScore: getenvFloat("RERANK_MIN_SCORE", defaultRerankMinScore),
		Vectors:        vector.New(getenv("QDRANT_URL", "http://localhost:6333"), os.Getenv("QDRANT_API_KEY")),
		LLM:            llm.New(getenv("LLM_BASE_URL", llm.DefaultBaseURL), groqKey, getenv("GROQ_MODEL", "openai/gpt-oss-120b")),
		Files:          files,

		MinScore:    getenvFloat("MIN_SCORE", rag.DefaultMinScore),
		ScoreMargin: getenvFloat("SCORE_MARGIN", rag.DefaultScoreMargin),

		ChatLimit:   limiter("CHAT_RATE_PER_MIN", 20, 10, trustProxy),
		IngestLimit: limiter("INGEST_RATE_PER_MIN", 5, 3, trustProxy),

		AdminToken: os.Getenv("ADMIN_TOKEN"),
	})
	if os.Getenv("ADMIN_TOKEN") == "" && staticDirSet() {
		log.Print("warning: ADMIN_TOKEN is not set; anyone can upload and delete documents")
	}

	staticDir := os.Getenv("STATIC_DIR")
	if staticDir != "" {
		log.Printf("serving frontend from %s, API under /api", staticDir)
	}
	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           server.New(h.Routes(), staticDir),
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: /chat streams for as long as the model takes.
	}

	log.Printf("API listening on :%s", port)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

// defaultRerankMinScore is tuned with cmd/eval -tune for bge-reranker-v2-m3.
const defaultRerankMinScore = 0.1

// staticDirSet reports production mode (the API serves the frontend).
func staticDirSet() bool { return os.Getenv("STATIC_DIR") != "" }

// limiter builds a per-client limiter from KEY (requests per minute; 0
// disables it). The burst lets a few requests through back to back.
func limiter(key string, perMinute, burst int, trustProxy bool) *ratelimit.Limiter {
	if v := os.Getenv(key); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			log.Fatalf("%s: invalid number %q", key, v)
		}
		perMinute = n
	}
	if perMinute == 0 {
		return nil
	}
	return ratelimit.New(perMinute, min(burst, perMinute), trustProxy)
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
