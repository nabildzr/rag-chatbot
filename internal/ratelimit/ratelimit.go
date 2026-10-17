// Package ratelimit is a small in-memory, per-client token bucket limiter.
// It suits a single API instance; several replicas would need a shared store.
package ratelimit

import (
	"encoding/json"
	"math"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

type bucket struct {
	tokens float64
	last   time.Time
}

type Limiter struct {
	perSecond  float64
	burst      float64
	trustProxy bool
	now        func() time.Time

	mu      sync.Mutex
	buckets map[string]*bucket
	calls   int
}

// New allows perMinute requests per client on average, with bursts of up to
// burst requests. Set trustProxy only behind a reverse proxy that overwrites
// X-Forwarded-For, otherwise clients can spoof their key.
func New(perMinute, burst int, trustProxy bool) *Limiter {
	return &Limiter{
		perSecond:  float64(perMinute) / 60,
		burst:      float64(max(burst, 1)),
		trustProxy: trustProxy,
		now:        time.Now,
		buckets:    map[string]*bucket{},
	}
}

// Allow takes one token for key. When none is left it reports how long until
// the next one is available.
func (l *Limiter) Allow(key string) (bool, time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()

	now := l.now()
	l.calls++
	if l.calls%1000 == 0 {
		l.prune(now)
	}

	b, ok := l.buckets[key]
	if !ok {
		b = &bucket{tokens: l.burst, last: now}
		l.buckets[key] = b
	}
	b.tokens = math.Min(l.burst, b.tokens+now.Sub(b.last).Seconds()*l.perSecond)
	b.last = now

	if b.tokens >= 1 {
		b.tokens--
		return true, 0
	}
	wait := time.Duration((1 - b.tokens) / l.perSecond * float64(time.Second))
	return false, wait
}

// prune drops buckets that have refilled completely; they behave exactly
// like new ones, so forgetting them changes nothing.
func (l *Limiter) prune(now time.Time) {
	full := time.Duration(l.burst / l.perSecond * float64(time.Second))
	for k, b := range l.buckets {
		if now.Sub(b.last) > full {
			delete(l.buckets, k)
		}
	}
}

// Wrap rejects requests over the limit with 429 and a Retry-After header.
// A nil Limiter lets everything through.
func (l *Limiter) Wrap(next http.HandlerFunc) http.HandlerFunc {
	if l == nil {
		return next
	}
	return func(w http.ResponseWriter, r *http.Request) {
		ok, wait := l.Allow(l.clientIP(r))
		if !ok {
			secs := int(math.Ceil(wait.Seconds()))
			w.Header().Set("Retry-After", strconv.Itoa(secs))
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			json.NewEncoder(w).Encode(map[string]string{
				"error": "Terlalu banyak permintaan. Coba lagi dalam " + strconv.Itoa(secs) + " detik.",
			})
			return
		}
		next(w, r)
	}
}

func (l *Limiter) clientIP(r *http.Request) string {
	if l.trustProxy {
		if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
			first, _, _ := strings.Cut(xff, ",")
			return strings.TrimSpace(first)
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
