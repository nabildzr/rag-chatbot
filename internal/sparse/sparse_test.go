package sparse

import (
	"reflect"
	"testing"
)

func TestTokenize(t *testing.T) {
	got := Tokenize("Berapa biaya Rp3.500 per km? GET /readyz returns OOMKilled, x y.")
	want := []string{"biaya", "rp3", "500", "per", "km", "get", "readyz", "returns", "oomkilled"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestDocumentSaturatesRepeats(t *testing.T) {
	once := Document("qdrant")
	many := Document("qdrant qdrant qdrant qdrant qdrant qdrant")
	if len(once.Indices) != 1 || once.Indices[0] != many.Indices[0] {
		t.Fatalf("same term must map to the same index: %v %v", once, many)
	}
	// More occurrences weigh more, but far less than linearly.
	if !(many.Values[0] > once.Values[0] && many.Values[0] < 3*once.Values[0]) {
		t.Fatalf("no saturation: once %.3f, six times %.3f", once.Values[0], many.Values[0])
	}
}

func TestQueryWeightsAreOne(t *testing.T) {
	q := Query("readyz readyz probe")
	if len(q.Indices) != 2 || q.Values[0] != 1 || q.Values[1] != 1 {
		t.Fatalf("got %+v", q)
	}
	for i := 1; i < len(q.Indices); i++ {
		if q.Indices[i-1] >= q.Indices[i] {
			t.Fatal("indices must be sorted and unique")
		}
	}
}

func TestContainsAll(t *testing.T) {
	text := "Waisak dipusatkan di Borobudur dengan prosesi dari Candi Mendut."
	for q, want := range map[string]bool{
		"Mendut":                     true,
		"candi mendut?":              true,
		"Berapa gaji direktur utama": false,
		"apa itu":                    false, // only stop words
	} {
		if got := ContainsAll(text, q); got != want {
			t.Errorf("%q: got %v", q, got)
		}
	}
}
