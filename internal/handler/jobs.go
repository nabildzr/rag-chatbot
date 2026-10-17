package handler

import (
	"errors"
	"sync"
	"time"

	"github.com/nabildzr/rag-chatbot/internal/vector"
)

// job tracks one background ingest. Jobs live in memory: they are only for
// progress reporting, the indexed data itself is in Qdrant.
type job struct {
	ID         string
	DocumentID string
	Filename   string
	hash       string

	mu       sync.Mutex
	status   string // queued, extracting, embedding, done, error
	done     int
	total    int
	errMsg   string
	result   *document
	updated  time.Time
	finished chan struct{}
}

type jobSnapshot struct {
	ID         string    `json:"job_id"`
	DocumentID string    `json:"document_id"`
	Filename   string    `json:"filename"`
	Status     string    `json:"status"`
	Done       int       `json:"done"`
	Total      int       `json:"total"`
	Error      string    `json:"error,omitempty"`
	Document   *document `json:"document,omitempty"`
}

func (j *job) snapshot() jobSnapshot {
	j.mu.Lock()
	defer j.mu.Unlock()
	return jobSnapshot{j.ID, j.DocumentID, j.Filename, j.status, j.done, j.total, j.errMsg, j.result}
}

type jobList struct {
	mu   sync.Mutex
	jobs map[string]*job
}

const jobRetention = time.Hour

// start registers a queued job. If a job for the same content is still
// running it returns that job's document ID instead.
func (l *jobList) start(docID, filename, hash string) (*job, string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.jobs == nil {
		l.jobs = map[string]*job{}
	}
	now := time.Now()
	for id, j := range l.jobs {
		j.mu.Lock()
		active := j.status != "done" && j.status != "error"
		stale := !active && now.Sub(j.updated) > jobRetention
		j.mu.Unlock()
		if active && j.hash == hash {
			return nil, j.DocumentID
		}
		if stale {
			delete(l.jobs, id)
		}
	}
	j := &job{
		ID: vector.NewID(), DocumentID: docID, Filename: filename, hash: hash,
		status: "queued", updated: now, finished: make(chan struct{}),
	}
	l.jobs[j.ID] = j
	return j, ""
}

func (l *jobList) get(id string) *job {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.jobs[id]
}

func (l *jobList) update(j *job, status string, done, total int) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.status, j.done, j.total, j.updated = status, done, total, time.Now()
}

func (l *jobList) finish(j *job, result *document) {
	j.mu.Lock()
	j.status, j.done, j.result, j.updated = "done", j.total, result, time.Now()
	j.mu.Unlock()
	close(j.finished)
}

// fail records err; only errors about the file itself are shown verbatim.
func (l *jobList) fail(j *job, err error) {
	msg := "pemrosesan gagal, coba lagi nanti"
	var ue userError
	if errors.As(err, &ue) {
		msg = ue.Error()
	}
	j.mu.Lock()
	j.status, j.errMsg, j.updated = "error", msg, time.Now()
	j.mu.Unlock()
	close(j.finished)
}
