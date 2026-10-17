// Package store keeps uploaded original files on disk, named by document ID,
// so they can be downloaded, cited and re-indexed later.
package store

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var ErrNotFound = errors.New("file not found")

// Document IDs are server-generated UUIDs; anything else is refused so an ID
// can never be used to reach outside the directory.
var validID = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

type Files struct {
	dir string
}

func New(dir string) (*Files, error) {
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, err
	}
	return &Files{dir: dir}, nil
}

// Save writes data as <id><ext>, atomically via a temporary file.
func (f *Files) Save(id, ext string, data []byte) error {
	if !validID.MatchString(id) {
		return fmt.Errorf("store: invalid id %q", id)
	}
	ext = strings.ToLower(filepath.Ext("x" + ext))
	tmp, err := os.CreateTemp(f.dir, ".upload-*")
	if err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	return os.Rename(tmp.Name(), filepath.Join(f.dir, id+ext))
}

// Path returns the stored file for id.
func (f *Files) Path(id string) (string, error) {
	if !validID.MatchString(id) {
		return "", ErrNotFound
	}
	matches, err := filepath.Glob(filepath.Join(f.dir, id+".*"))
	if err != nil {
		return "", err
	}
	if len(matches) == 0 {
		return "", ErrNotFound
	}
	return matches[0], nil
}

// Delete removes the file for id; a missing file is not an error.
func (f *Files) Delete(id string) error {
	p, err := f.Path(id)
	if errors.Is(err, ErrNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	return os.Remove(p)
}
