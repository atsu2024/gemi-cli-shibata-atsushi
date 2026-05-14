package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// BankData represents the input information
type BankData struct {
	Name    string
	Account string
}

// sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// processThroughDNN simulates a Deep Neural Network forward pass for a specific bank entry using goroutines
func processThroughDNN(wg *sync.WaitGroup, bank BankData, layerID int, resultChan chan<- string) {
	defer wg.Done()

	// Simulate some complex DNN processing time
	time.Sleep(time.Duration(rand.Intn(500)) * time.Millisecond)

	// Mock high-precision calculation (DNN weights/biases)
	// In a real scenario, this would involve matrix multiplication
	inputVal := float64(len(bank.Name) + len(bank.Account))
	weight := 0.523456789123456 // High precision simulation
	bias := 0.123456789

	// Forward pass: Hidden Layer
	hiddenValue := sigmoid(inputVal*weight + bias)

	// Final Output Layer
	finalScore := sigmoid(hiddenValue * 0.888)

	resultChan <- fmt.Sprintf(
		"[%s] DNN Layer %d Processing Complete. Account: %s | Score: %.15f",
		bank.Name, layerID, bank.Account, finalScore,
	)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== High-Precision Deep Learning Simulation (Goroutine Edition) ===")
	fmt.Println("Initializing DNN Processing for Bank Information...")

	banks := []BankData{
		{"Saitama Risona", "793-0-4366399"},
		{"Mitsui Sumitomo", "200-4902647"},
	}

	var wg sync.WaitGroup
	resultChan := make(chan string, len(banks)*3) // 3 layers per bank

	fmt.Println("Launching Goroutines for Parallel Deep Learning Inference...")

	// Launch parallel processing for each bank data through multiple simulated DNN layers
	for _, bank := range banks {
		for layer := 1; layer <= 3; layer++ {
			wg.Add(1)
			go processThroughDNN(&wg, bank, layer, resultChan)
		}
	}

	// Close channel once all goroutines finish
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect and display high-precision results
	for res := range resultChan {
		fmt.Println(res)
	}

	fmt.Println("\nDeep Learning (DNN) Simulation for specific datasets is complete.")
}
