// Command eval measures retrieval quality against a running API.
//
// It uploads the files in eval/corpus, asks every question in
// eval/questions.json through POST /search (retrieval only, no LLM) across
// all documents, and reports how often the expected section comes back:
//
//	go run ./cmd/eval -api http://localhost:8080
//
// Question types: "same" (question in the document's language), "cross"
// (the other language) and "offtopic" (nothing should be returned).
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"
)

type question struct {
	Q       string `json:"q"`
	File    string `json:"file"`
	Section int    `json:"section"`
	Type    string `json:"type"`
}

type source struct {
	Filename string  `json:"filename"`
	Page     int     `json:"page"`
	Score    float32 `json:"score"`
}

var (
	api     = flag.String("api", "http://localhost:8080", "API base URL (with /api for a production build)")
	corpus  = flag.String("corpus", "eval/corpus", "directory of documents to upload")
	qfile   = flag.String("questions", "eval/questions.json", "questions with expected sections")
	keep    = flag.Bool("keep", false, "keep uploaded documents afterwards")
	verbose = flag.Bool("v", false, "print every miss")
	token   = os.Getenv("ADMIN_TOKEN")
)

func main() {
	flag.Parse()
	log.SetFlags(0)

	var questions []question
	raw, err := os.ReadFile(*qfile)
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(raw, &questions); err != nil {
		log.Fatalf("%s: %v", *qfile, err)
	}

	files, _ := filepath.Glob(filepath.Join(*corpus, "*"))
	var created []string
	for _, f := range files {
		id, isNew, err := ingest(f)
		if err != nil {
			log.Fatalf("ingest %s: %v", f, err)
		}
		if isNew {
			created = append(created, id)
		}
		log.Printf("indexed %s", filepath.Base(f))
	}
	defer func() {
		if *keep {
			return
		}
		for _, id := range created {
			req, _ := http.NewRequest(http.MethodDelete, *api+"/documents/"+id, nil)
			authorize(req)
			if resp, err := http.DefaultClient.Do(req); err == nil {
				resp.Body.Close()
			}
		}
	}()

	type stats struct {
		n, hit1, hit5 int
		rr            float64
	}
	byType := map[string]*stats{}
	start := time.Now()
	for _, q := range questions {
		sources, err := search(q.Q)
		if err != nil {
			log.Fatalf("search %q: %v", q.Q, err)
		}
		s := byType[q.Type]
		if s == nil {
			s = &stats{}
			byType[q.Type] = s
		}
		s.n++

		if q.Type == "offtopic" {
			if len(sources) == 0 {
				s.hit1++
				s.hit5++
				s.rr++
			} else if *verbose {
				log.Printf("  FALSE HIT  %q -> %s p%d (%.2f)", q.Q, sources[0].Filename, sources[0].Page, sources[0].Score)
			}
			continue
		}

		rank := 0
		for i, src := range sources {
			if src.Filename == q.File && src.Page == q.Section {
				rank = i + 1
				break
			}
		}
		if rank == 1 {
			s.hit1++
		}
		if rank > 0 && rank <= 5 {
			s.hit5++
			s.rr += 1 / float64(rank)
		}
		if rank != 1 && *verbose {
			got := "nothing"
			if len(sources) > 0 {
				got = fmt.Sprintf("%s p%d (%.2f)", sources[0].Filename, sources[0].Page, sources[0].Score)
			}
			log.Printf("  MISS(rank %d) [%s] %q -> want %s p%d, top %s", rank, q.Type, q.Q, q.File, q.Section, got)
		}
	}

	types := make([]string, 0, len(byType))
	for t := range byType {
		types = append(types, t)
	}
	sort.Strings(types)
	fmt.Printf("\n| type | n | hit@1 | hit@5 | MRR |\n|---|---|---|---|---|\n")
	var total stats
	for _, t := range types {
		s := byType[t]
		if t == "offtopic" {
			fmt.Printf("| offtopic (no sources) | %d | %.0f%% | | |\n", s.n, pct(s.hit1, s.n))
			continue
		}
		fmt.Printf("| %s | %d | %.0f%% | %.0f%% | %.2f |\n", t, s.n, pct(s.hit1, s.n), pct(s.hit5, s.n), s.rr/float64(s.n))
		total.n += s.n
		total.hit1 += s.hit1
		total.hit5 += s.hit5
		total.rr += s.rr
	}
	fmt.Printf("| **in-scope total** | %d | %.0f%% | %.0f%% | %.2f |\n", total.n, pct(total.hit1, total.n), pct(total.hit5, total.n), total.rr/float64(total.n))
	fmt.Printf("\n%d questions in %s\n", len(questions), time.Since(start).Round(time.Second))
}

func pct(a, n int) float64 { return 100 * float64(a) / float64(max(n, 1)) }

func authorize(req *http.Request) {
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
}

// do sends a request, waiting out 429 responses as Retry-After says.
func do(build func() *http.Request) (*http.Response, error) {
	for {
		resp, err := http.DefaultClient.Do(build())
		if err != nil || resp.StatusCode != http.StatusTooManyRequests {
			return resp, err
		}
		resp.Body.Close()
		wait, _ := strconv.Atoi(resp.Header.Get("Retry-After"))
		time.Sleep(time.Duration(max(wait, 1)) * time.Second)
	}
}

// ingest uploads path and waits for indexing. isNew is false when the same
// file was already in the index (409).
func ingest(path string) (id string, isNew bool, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", false, err
	}
	resp, err := do(func() *http.Request {
		var body bytes.Buffer
		mw := multipart.NewWriter(&body)
		fw, _ := mw.CreateFormFile("file", filepath.Base(path))
		fw.Write(data)
		mw.Close()
		req, _ := http.NewRequest(http.MethodPost, *api+"/ingest", &body)
		req.Header.Set("Content-Type", mw.FormDataContentType())
		authorize(req)
		return req
	})
	if err != nil {
		return "", false, err
	}
	defer resp.Body.Close()
	var acc struct {
		JobID      string `json:"job_id"`
		DocumentID string `json:"document_id"`
		Error      string `json:"error"`
	}
	json.NewDecoder(resp.Body).Decode(&acc)
	switch resp.StatusCode {
	case http.StatusConflict:
		return acc.DocumentID, false, nil
	case http.StatusAccepted:
	default:
		return "", false, fmt.Errorf("status %d: %s", resp.StatusCode, acc.Error)
	}

	for {
		time.Sleep(300 * time.Millisecond)
		r, err := http.Get(*api + "/ingest/" + acc.JobID)
		if err != nil {
			return "", false, err
		}
		var job struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		json.NewDecoder(r.Body).Decode(&job)
		r.Body.Close()
		switch job.Status {
		case "done":
			return acc.DocumentID, true, nil
		case "error", "":
			return "", false, fmt.Errorf("job failed: %s", job.Error)
		}
	}
}

func search(q string) ([]source, error) {
	payload, _ := json.Marshal(map[string]string{"question": q})
	resp, err := do(func() *http.Request {
		req, _ := http.NewRequest(http.MethodPost, *api+"/search", bytes.NewReader(payload))
		req.Header.Set("Content-Type", "application/json")
		return req
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, b)
	}
	var res struct {
		Sources []source `json:"sources"`
	}
	err = json.NewDecoder(resp.Body).Decode(&res)
	return res.Sources, err
}
