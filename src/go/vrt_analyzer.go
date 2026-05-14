package main

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// FileResult stores the analysis results for a single file
type FileResult struct {
	Path string
	Size int64
	Hash string
	Err  error
}

func analyzeFile(path string, wg *sync.WaitGroup, results chan<- FileResult) {
	defer wg.Done()

	file, err := os.Open(path)
	if err != nil {
		results <- FileResult{Path: path, Err: err}
		return
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		results <- FileResult{Path: path, Err: err}
		return
	}

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		results <- FileResult{Path: path, Err: err}
		return
	}

	results <- FileResult{
		Path: path,
		Size: info.Size(),
		Hash: fmt.Sprintf("%x", hash.Sum(nil)),
	}
}

func main() {
	fmt.Println("VRT Multi-File Analyzer (Goroutine Parallelism)")
	fmt.Println("Analyzing VipreRemovalTool executable files...\n")

	// Find all VipreRemovalTool*.exe files
	files, err := filepath.Glob("VipreRemovalTool*.exe")
	if err != nil {
		fmt.Printf("Error finding files: %v\n", err)
		return
	}

	if len(files) == 0 {
		fmt.Println("No VipreRemovalTool*.exe files found in the current directory.")
		return
	}

	start := time.Now()
	results := make(chan FileResult, len(files))
	var wg sync.WaitGroup

	for _, file := range files {
		wg.Add(1)
		go analyzeFile(file, &wg, results)
	}

	// Close results channel when all goroutines are done
	go func() {
		wg.Wait()
		close(results)
	}()

	fmt.Printf("%-30s | %-12s | %s\n", "Filename", "Size (Bytes)", "SHA-256 Hash")
	fmt.Println("------------------------------------------------------------------------------------------")

	for res := range results {
		if res.Err != nil {
			fmt.Printf("%-30s | Error: %v\n", filepath.Base(res.Path), res.Err)
		} else {
			fmt.Printf("%-30s | %-12d | %s...\n", filepath.Base(res.Path), res.Size, res.Hash[:16])
		}
	}

	fmt.Println("------------------------------------------------------------------------------------------")
	fmt.Printf("Analysis completed in %v\n", time.Since(start))
}
