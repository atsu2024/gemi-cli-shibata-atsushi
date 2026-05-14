package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// MathFunction represents a mathematical function parameter set from MathDraw
type MathFunction struct {
	Name string
	A    float64
	B    float64
	Mode string
}

// sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// processThroughDNN simulates a Deep Neural Network forward pass that "learns" or evaluates a function
func processThroughDNN(wg *sync.WaitGroup, fn MathFunction, layerID int, resultChan chan<- string) {
	defer wg.Done()

	// Simulate complex DNN processing time
	time.Sleep(time.Duration(rand.Intn(300)) * time.Millisecond)

	// In a real DNN, weights and biases would be learned. 
	// Here we simulate high-precision weight application.
	// We use the function parameters (A, B) as inputs to the network.
	inputVal := fn.A*1.5 + fn.B*0.5
	weight := 0.7654321098765432 // High precision weight
	bias := -0.2345678901234567  // High precision bias

	// Hidden Layer 1
	h1 := sigmoid(inputVal*weight + bias)

	// Hidden Layer 2 (simulated)
	h2 := sigmoid(h1*1.234 - 0.567)

	// Output Layer
	finalOutput := sigmoid(h2 * 0.999)

	resultChan <- fmt.Sprintf(
		"[%-10s] Mode: %-10s | A: %12.4f | B: %12.4f | DNN Layer %d Output: %.18f",
		fn.Name, fn.Mode, fn.A, fn.B, layerID, finalOutput,
	)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("======================================================================")
	fmt.Println("   High-Precision MathDraw DNN Simulation (Goroutine Parallelism)    ")
	fmt.Println("======================================================================")
	fmt.Println("Initializing DNN processing for mathematical functions...")

	// Data derived from MathDraw.py logic (y=ax+b, y=axx+b, etc.)
	functions := []MathFunction{
		{"Line 1", 2.5, 10.0, "y=ax+b"},
		{"Line 2", -1.2, 5.5, "y=ax+b"},
		{"Quad 1", 0.5, -2.0, "y=axx+b"},
		{"Quad 2", -0.8, 15.0, "y=axx+b"},
		{"Hyper 1", 100.0, 0.0, "xy=a"},
	}

	var wg sync.WaitGroup
	// We'll process each function through 4 simulated DNN layers in parallel
	numLayers := 4
	resultChan := make(chan string, len(functions)*numLayers)

	fmt.Printf("Launching %d goroutines for parallel Deep Learning inference...\n\n", len(functions)*numLayers)

	startTime := time.Now()

	for _, fn := range functions {
		for layer := 1; layer <= numLayers; layer++ {
			wg.Add(1)
			go processThroughDNN(&wg, fn, layer, resultChan)
		}
	}

	// Wait for all goroutines in a separate closer
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	for res := range resultChan {
		fmt.Println(res)
	}

	duration := time.Since(startTime)
	fmt.Println("----------------------------------------------------------------------")
	fmt.Printf("Deep Learning (DNN) processing completed in %v\n", duration)
	fmt.Println("All mathematical function representations have been simulated.")
}
