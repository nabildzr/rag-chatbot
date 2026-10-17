package ratelimit

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func fakeClock(l *Limiter) *time.Time {
	t := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	l.now = func() time.Time { return t }
	return &t
}

func TestBurstThenRefill(t *testing.T) {
	l := New(60, 3, false) // one token per second
	clock := fakeClock(l)

	for i := 0; i < 3; i++ {
		if ok, _ := l.Allow("a"); !ok {
			t.Fatalf("request %d within burst was rejected", i+1)
		}
	}
	ok, wait := l.Allow("a")
	if ok || wait != time.Second {
		t.Fatalf("4th request: ok=%v wait=%v, want rejected with 1s wait", ok, wait)
	}
	if ok, _ := l.Allow("b"); !ok {
		t.Fatal("other client should have its own bucket")
	}

	*clock = clock.Add(time.Second)
	if ok, _ := l.Allow("a"); !ok {
		t.Fatal("token should refill after 1s")
	}
}

func TestPruneForgetsFullBuckets(t *testing.T) {
	l := New(60, 2, false)
	clock := fakeClock(l)
	l.Allow("a")
	*clock = clock.Add(time.Minute)
	l.prune(*clock)
	if len(l.buckets) != 0 {
		t.Fatalf("want empty, got %d buckets", len(l.buckets))
	}
}

func TestWrapReturns429(t *testing.T) {
	l := New(1, 1, false)
	h := l.Wrap(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusNoContent) })

	do := func(remote, xff string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/chat", nil)
		req.RemoteAddr = remote
		if xff != "" {
			req.Header.Set("X-Forwarded-For", xff)
		}
		rec := httptest.NewRecorder()
		h(rec, req)
		return rec
	}

	if rec := do("1.2.3.4:5000", ""); rec.Code != http.StatusNoContent {
		t.Fatalf("first: %d", rec.Code)
	}
	rec := do("1.2.3.4:6000", "9.9.9.9") // same IP, new port; XFF ignored without trustProxy
	if rec.Code != http.StatusTooManyRequests || rec.Header().Get("Retry-After") != "60" {
		t.Fatalf("second: %d retry-after=%q", rec.Code, rec.Header().Get("Retry-After"))
	}
}

func TestTrustProxyUsesForwardedFor(t *testing.T) {
	l := New(60, 1, true)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("X-Forwarded-For", "203.0.113.7, 10.0.0.1")
	if got := l.clientIP(req); got != "203.0.113.7" {
		t.Fatalf("got %q", got)
	}
}

func TestNilLimiterPassesThrough(t *testing.T) {
	var l *Limiter
	called := false
	l.Wrap(func(http.ResponseWriter, *http.Request) { called = true })(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/", nil))
	if !called {
		t.Fatal("nil limiter blocked the request")
	}
}
