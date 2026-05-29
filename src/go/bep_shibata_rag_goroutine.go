package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
)

const author = "柴田敦史"

type RAGDocument struct {
	ID      int       `json:"id"`
	Content string    `json:"content"`
	Vector  []float64 `json:"vector"`
}

type SearchHit struct {
	Document RAGDocument
	Score    float64
}

type BEPInput struct {
	Name         string
	FixedCost    float64
	VariableCost float64
	SellingPrice float64
	ActualQty    float64
}

type BEPResult struct {
	Input                   BEPInput
	ContributionMargin      float64
	ContributionMarginRatio float64
	BreakEvenQuantity       float64
	BreakEvenSales          float64
	ActualSales             float64
	Profit                  float64
	SafetyMargin            float64
	SafetyMarginRatio       float64
}

func queryVector(query string) []float64 {
	terms := []string{
		"dnn", "precision", "break-even", "cost", "data",
		"high-precision", "goroutine", "simulation", "finance", "rag",
	}
	lower := strings.ToLower(query)
	vector := make([]float64, len(terms))
	for i, term := range terms {
		if strings.Contains(lower, term) {
			vector[i] = 1.0
		}
	}
	if strings.Contains(query, "損益分岐点") || strings.Contains(query, "BEP") {
		vector[2] = 1.0
		vector[8] = 0.9
	}
	if strings.Contains(query, "goroutine") || strings.Contains(query, "ゴルーチン") {
		vector[6] = 1.0
	}
	if strings.Contains(strings.ToLower(query), "rag") {
		vector[9] = 1.0
	}
	return vector
}

func cosine(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func loadRAG(path string) []RAGDocument {
	data, err := os.ReadFile(path)
	if err != nil {
		return fallbackRAG()
	}

	var docs []RAGDocument
	if err := json.Unmarshal(data, &docs); err != nil {
		return fallbackRAG()
	}
	return docs
}

func fallbackRAG() []RAGDocument {
	return []RAGDocument{
		{ID: 1, Content: "Break-even point analysis: fixed cost, variable cost, selling price, BEP quantity, BEP sales, profit, and safety margin.", Vector: []float64{0.45, 0.35, 1.00, 0.10, 0.30, 0.25, 0.35, 0.05, 0.80, 0.10}},
		{ID: 2, Content: "Go goroutine parallel processing: use sync.WaitGroup and channels to split calculation work across CPU cores.", Vector: []float64{0.85, 0.25, 0.05, 0.20, 0.10, 0.20, 1.00, 0.05, 0.20, 0.15}},
		{ID: 3, Content: "RAG retrieval engine: rank local documents by cosine similarity and pass the best context into analysis.", Vector: []float64{0.20, 0.70, 0.10, 0.20, 0.90, 0.75, 0.05, 0.20, 0.70, 0.90}},
	}
}

func retrieve(docs []RAGDocument, query string, topK int) []SearchHit {
	qv := queryVector(query)
	workers := runtime.NumCPU()
	if workers > len(docs) {
		workers = len(docs)
	}
	if workers < 1 {
		workers = 1
	}

	hits := make([]SearchHit, 0, len(docs))
	ch := make(chan SearchHit, len(docs))
	var wg sync.WaitGroup
	chunk := (len(docs) + workers - 1) / workers

	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if start >= len(docs) {
			break
		}
		if end > len(docs) {
			end = len(docs)
		}

		wg.Add(1)
		go func(part []RAGDocument) {
			defer wg.Done()
			for _, doc := range part {
				score := cosine(qv, doc.Vector)
				if strings.Contains(strings.ToLower(doc.Content), "break-even") {
					score += 0.15
				}
				if strings.Contains(strings.ToLower(doc.Content), "goroutine") {
					score += 0.10
				}
				ch <- SearchHit{Document: doc, Score: score}
			}
		}(docs[start:end])
	}

	wg.Wait()
	close(ch)
	for hit := range ch {
		hits = append(hits, hit)
	}

	sort.Slice(hits, func(i, j int) bool {
		return hits[i].Score > hits[j].Score
	})
	if topK > len(hits) {
		topK = len(hits)
	}
	return hits[:topK]
}

func calculateBEP(input BEPInput) (BEPResult, error) {
	cm := input.SellingPrice - input.VariableCost
	if cm <= 0 {
		return BEPResult{}, fmt.Errorf("%s: 販売単価は変動費より大きい必要があります", input.Name)
	}
	if input.SellingPrice == 0 {
		return BEPResult{}, fmt.Errorf("%s: 販売単価が0です", input.Name)
	}

	actualSales := input.SellingPrice * input.ActualQty
	bepQty := input.FixedCost / cm
	bepSales := bepQty * input.SellingPrice
	profit := actualSales - (input.FixedCost + input.VariableCost*input.ActualQty)
	safety := actualSales - bepSales
	safetyRatio := 0.0
	if actualSales != 0 {
		safetyRatio = safety / actualSales * 100.0
	}

	return BEPResult{
		Input:                   input,
		ContributionMargin:      cm,
		ContributionMarginRatio: cm / input.SellingPrice * 100.0,
		BreakEvenQuantity:       bepQty,
		BreakEvenSales:          bepSales,
		ActualSales:             actualSales,
		Profit:                  profit,
		SafetyMargin:            safety,
		SafetyMarginRatio:       safetyRatio,
	}, nil
}

func calculateAll(inputs []BEPInput) ([]BEPResult, []error) {
	results := make([]BEPResult, len(inputs))
	errors := make([]error, len(inputs))
	var wg sync.WaitGroup

	for i, input := range inputs {
		wg.Add(1)
		go func(idx int, item BEPInput) {
			defer wg.Done()
			results[idx], errors[idx] = calculateBEP(item)
		}(i, input)
	}

	wg.Wait()
	return results, errors
}

func printHits(hits []SearchHit) {
	fmt.Println("[RAG retrieved context]")
	for _, hit := range hits {
		content := hit.Document.Content
		if len(content) > 96 {
			content = content[:96] + "..."
		}
		fmt.Printf("  id=%02d score=%.3f %s\n", hit.Document.ID, hit.Score, content)
	}
}

func printResults(results []BEPResult, errors []error) {
	fmt.Println("\n[Break-even results]")
	for i, err := range errors {
		if err != nil {
			fmt.Printf("  error: %v\n", err)
			continue
		}

		r := results[i]
		status := "黒字"
		if r.Profit < 0 {
			status = "赤字"
		} else if math.Abs(r.Profit) < 1e-9 {
			status = "損益分岐点"
		}

		fmt.Printf("  %s\n", r.Input.Name)
		fmt.Printf("    限界利益: %.2f / 限界利益率: %.2f%%\n", r.ContributionMargin, r.ContributionMarginRatio)
		fmt.Printf("    損益分岐点数量: %.2f / 損益分岐点売上高: %.2f\n", r.BreakEvenQuantity, r.BreakEvenSales)
		fmt.Printf("    実売上高: %.2f / 利益: %.2f / 安全余裕率: %.2f%% / 判定: %s\n",
			r.ActualSales, r.Profit, r.SafetyMarginRatio, status)
	}
}

func main() {
	fmt.Println("============================================================")
	fmt.Println("損益分岐点 RAG goroutine analyzer")
	fmt.Printf("author: %s\n", author)
	fmt.Printf("cpus: %d\n", runtime.NumCPU())
	fmt.Println("============================================================")

	ragPath := filepath.Join("RAG.json")
	docs := loadRAG(ragPath)
	hits := retrieve(docs, "損益分岐点 柴田敦史 RAG goroutine break-even finance", 3)
	printHits(hits)

	inputs := []BEPInput{
		{Name: "標準製品", FixedCost: 5000000, VariableCost: 3000, SellingPrice: 5000, ActualQty: 1500},
		{Name: "高付加価値製品", FixedCost: 12000000, VariableCost: 8000, SellingPrice: 20000, ActualQty: 1500},
		{Name: "量産製品", FixedCost: 800000, VariableCost: 150, SellingPrice: 200, ActualQty: 10000},
		{Name: "検証ケース", FixedCost: 5000, VariableCost: 50, SellingPrice: 200, ActualQty: 500},
	}

	results, errors := calculateAll(inputs)
	printResults(results, errors)

	fmt.Println("\nGo build target example:")
	fmt.Println("  go build -o bin/bep_shibata_rag_goroutine.exe src/go/bep_shibata_rag_goroutine.go")
}
