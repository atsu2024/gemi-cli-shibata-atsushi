package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// BEP DNN Parameters
const (
	InputSize    = 4  // FixedCost, VariableCost, SellingPrice, Qty
	HiddenSize1  = 16
	HiddenSize2  = 8
	OutputSize   = 1  // Profit
	LearningRate = 0.01
	Epochs       = 2000
	TrainSize    = 100
)

// Activation function: Sigmoid
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of sigmoid
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

type Neuron struct {
	weights []float64
	bias    float64
	output  float64
	delta   float64
}

type Layer struct {
	neurons []*Neuron
}

type DNN struct {
	layers []*Layer
}

func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())

	// Hidden Layer 1
	h1 := &Layer{neurons: make([]*Neuron, HiddenSize1)}
	for i := 0; i < HiddenSize1; i++ {
		n := &Neuron{weights: make([]float64, InputSize)}
		for j := 0; j < InputSize; j++ {
			n.weights[j] = rand.NormFloat64() * math.Sqrt(2.0/float64(InputSize))
		}
		h1.neurons[i] = n
	}

	// Hidden Layer 2
	h2 := &Layer{neurons: make([]*Neuron, HiddenSize2)}
	for i := 0; i < HiddenSize2; i++ {
		n := &Neuron{weights: make([]float64, HiddenSize1)}
		for j := 0; j < HiddenSize1; j++ {
			n.weights[j] = rand.NormFloat64() * math.Sqrt(2.0/float64(HiddenSize1))
		}
		h2.neurons[i] = n
	}

	// Output Layer
	out := &Layer{neurons: make([]*Neuron, OutputSize)}
	for i := 0; i < OutputSize; i++ {
		n := &Neuron{weights: make([]float64, HiddenSize2)}
		for j := 0; j < HiddenSize2; j++ {
			n.weights[j] = rand.NormFloat64() * math.Sqrt(2.0/float64(HiddenSize2))
		}
		out.neurons[i] = n
	}

	return &DNN{layers: []*Layer{h1, h2, out}}
}

// Forward pass with Goroutines
func (net *DNN) Forward(input []float64) []float64 {
	currentInput := input
	for _, layer := range net.layers {
		nextInput := make([]float64, len(layer.neurons))
		var wg sync.WaitGroup
		wg.Add(len(layer.neurons))

		for i, neuron := range layer.neurons {
			go func(idx int, n *Neuron, in []float64) {
				defer wg.Done()
				sum := n.bias
				for k, w := range n.weights {
					sum += w * in[k]
				}
				n.output = sigmoid(sum)
				nextInput[idx] = n.output
			}(i, neuron, currentInput)
		}
		wg.Wait()
		currentInput = nextInput
	}
	return currentInput
}

// Training with Backpropagation and Goroutines
func (net *DNN) Train(input, target []float64) {
	net.Forward(input)

	// 1. Output layer deltas
	outLayer := net.layers[2]
	for i, n := range outLayer.neurons {
		n.delta = (target[i] - n.output) * sigmoidDerivative(n.output)
	}

	// 2. Hidden layer 2 deltas
	h2Layer := net.layers[1]
	for i, n := range h2Layer.neurons {
		var errorSum float64
		for _, nextN := range outLayer.neurons {
			errorSum += nextN.delta * nextN.weights[i]
		}
		n.delta = errorSum * sigmoidDerivative(n.output)
	}

	// 3. Hidden layer 1 deltas
	h1Layer := net.layers[0]
	for i, n := range h1Layer.neurons {
		var errorSum float64
		for _, nextN := range h2Layer.neurons {
			errorSum += nextN.delta * nextN.weights[i]
		}
		n.delta = errorSum * sigmoidDerivative(n.output)
	}

	// 4. Update weights (Parallel)
	for lIdx, layer := range net.layers {
		var prevOutput []float64
		if lIdx == 0 {
			prevOutput = input
		} else {
			prevOutput = make([]float64, len(net.layers[lIdx-1].neurons))
			for i, n := range net.layers[lIdx-1].neurons {
				prevOutput[i] = n.output
			}
		}

		var wg sync.WaitGroup
		wg.Add(len(layer.neurons))
		for _, n := range layer.neurons {
			go func(neuron *Neuron, in []float64) {
				defer wg.Done()
				for i := range neuron.weights {
					neuron.weights[i] += LearningRate * neuron.delta * in[i]
				}
				neuron.bias += LearningRate * neuron.delta
			}(n, prevOutput)
		}
		wg.Wait()
	}
}

// BEP Logic for Data Generation
func calculateProfit(fixed, variable, price, qty float64) float64 {
	return qty*(price-variable) - fixed
}

// Normalization Helper
func normalize(val, min, max float64) float64 {
	if max == min {
		return 0.5
	}
	res := (val - min) / (max - min)
	if res < 0 { return 0 }
	if res > 1 { return 1 }
	return res
}

func main() {
	fmt.Println("================================================================")
	fmt.Println("  MathP: BEP Analysis using Deep Neural Network (DNN)")
	fmt.Println("  Parallel Execution with Go Goroutines")
	fmt.Println("================================================================")

	// Data Ranges
	minF, maxF := 1000.0, 10000.0
	minV, maxV := 10.0, 100.0
	minP, maxP := 110.0, 500.0
	minQ, maxQ := 1.0, 1000.0
	// Profit range: minQ*(minP-maxV)-maxF to maxQ*(maxP-minV)-minF
	minProfit := 1.0*(110.0-100.0) - 10000.0 // -9990
	maxProfit := 1000.0*(500.0-10.0) - 1000.0 // 489000

	// Generate Training Data
	inputs := make([][]float64, TrainSize)
	targets := make([][]float64, TrainSize)

	for i := 0; i < TrainSize; i++ {
		f := rand.Float64()*(maxF-minF) + minF
		v := rand.Float64()*(maxV-minV) + minV
		p := rand.Float64()*(maxP-minP) + minP
		q := rand.Float64()*(maxQ-minQ) + minQ

		profit := calculateProfit(f, v, p, q)

		inputs[i] = []float64{
			normalize(f, minF, maxF),
			normalize(v, minV, maxV),
			normalize(p, minP, maxP),
			normalize(q, minQ, maxQ),
		}
		targets[i] = []float64{normalize(profit, minProfit, maxProfit)}
	}

	dnn := NewDNN()

	fmt.Printf("Training started (%d epochs, %d samples)...\n", Epochs, TrainSize)
	start := time.Now()
	for e := 1; e <= Epochs; e++ {
		mse := 0.0
		for i := 0; i < TrainSize; i++ {
			dnn.Train(inputs[i], targets[i])
			if e%100 == 0 {
				out := dnn.Forward(inputs[i])
				mse += math.Pow(targets[i][0]-out[0], 2)
			}
		}
		if e%100 == 0 {
			fmt.Printf("Epoch %d/%d - MSE: %.6f\n", e, Epochs, mse/float64(TrainSize))
		}
	}
	fmt.Printf("Training finished in %v\n", time.Since(start))

	// Test Case
	fmt.Println("\n--- Test Prediction ---")
	testF, testV, testP, testQ := 5000.0, 50.0, 200.0, 500.0
	realProfit := calculateProfit(testF, testV, testP, testQ)
	
	testInput := []float64{
		normalize(testF, minF, maxF),
		normalize(testV, minV, maxV),
		normalize(testP, minP, maxP),
		normalize(testQ, minQ, maxQ),
	}
	
	predNorm := dnn.Forward(testInput)[0]
	predProfit := predNorm*(maxProfit-minProfit) + minProfit

	fmt.Printf("Input: Fixed=%.0f, Var=%.0f, Price=%.0f, Qty=%.0f\n", testF, testV, testP, testQ)
	fmt.Printf("Real Profit:      %10.2f\n", realProfit)
	fmt.Printf("Predicted Profit: %10.2f\n", predProfit)
	fmt.Printf("Accuracy:         %10.2f%%\n", (1-math.Abs(realProfit-predProfit)/math.Abs(maxProfit-minProfit))*100)
	
	fmt.Println("\n================================================================")
}
