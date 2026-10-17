package store

import (
	"errors"
	"os"
	"testing"
)

const id = "3f2c1b9a-1d2e-4f5a-8b6c-7d8e9f0a1b2c"

func TestSavePathDelete(t *testing.T) {
	f, err := New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Save(id, ".PDF", []byte("%PDF")); err != nil {
		t.Fatal(err)
	}
	p, err := f.Path(id)
	if err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(p); string(b) != "%PDF" {
		t.Fatalf("got %q", b)
	}
	if err := f.Delete(id); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Path(id); !errors.Is(err, ErrNotFound) {
		t.Fatalf("after delete: %v", err)
	}
	if err := f.Delete(id); err != nil {
		t.Fatalf("deleting twice: %v", err)
	}
}

func TestRejectsUnsafeIDs(t *testing.T) {
	f, _ := New(t.TempDir())
	for _, bad := range []string{"../etc/passwd", "*", "", "abc"} {
		if err := f.Save(bad, ".txt", nil); err == nil {
			t.Errorf("Save(%q) accepted", bad)
		}
		if _, err := f.Path(bad); !errors.Is(err, ErrNotFound) {
			t.Errorf("Path(%q): %v", bad, err)
		}
	}
}
