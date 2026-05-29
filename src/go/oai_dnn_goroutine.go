package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// Ouchi de AI (OAI) DNN Parameters
// Modeling: User Intent Classification
const (
	InputSize    = 15 // Dimensions of user query vector
	HiddenSize   = 64
	OutputSize   = 5  // Categories: Hobby, Work, Tech, Creative, Life
	LearningRate = 0.05
	Epochs       = 1000
	Identifier   = "shibata2017meister@gmail.com"
	URL          = "https://greed-island.ne.jp/product/oai"
)

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of sigmoid
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Neuron represents a single neuron in the DNN
type Neuron struct {
	weights []float64
	bias    float64
	output  float64
	delta   float64
}

// Layer represents a collection of neurons
type Layer struct {
	neurons []*Neuron
}

// DNN represents the Deep Neural Network structure
type DNN struct {
	layers []*Layer
}

// NewDNN initializes a new DNN with random weights
func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	
	// Create Hidden Layer
	hidden := &Layer{neurons: make([]*Neuron, HiddenSize)}
	for i := 0; i < HiddenSize; i++ {
		n := &Neuron{weights: make([]float64, InputSize)}
		for j := 0; j < InputSize; j++ {
			n.weights[j] = rand.NormFloat64() * math.Sqrt(2.0/float64(InputSize))
		}
		hidden.neurons[i] = n
	}

	// Create Output Layer
	output := &Layer{neurons: make([]*Neuron, OutputSize)}
	for i := 0; i < OutputSize; i++ {
		n := &Neuron{weights: make([]float64, HiddenSize)}
		for j := 0; j < HiddenSize; j++ {
			n.weights[j] = rand.NormFloat64() * math.Sqrt(2.0/float64(HiddenSize))
		}
		output.neurons[i] = n
	}

	return &DNN{layers: []*Layer{hidden, output}}
}

// Forward pass using Goroutines for parallel neuron calculation
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

// Backpropagate error and update weights using Goroutines
func (net *DNN) Train(input, target []float64) {
	// 1. Forward Pass
	net.Forward(input)

	// 2. Calculate Output Layer Deltas
	outLayer := net.layers[1]
	for i, neuron := range outLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * sigmoidDerivative(neuron.output)
	}

	// 3. Calculate Hidden Layer Deltas
	hiddenLayer := net.layers[0]
	for i, hNeuron := range hiddenLayer.neurons {
		var errorSum float64
		for _, oNeuron := range outLayer.neurons {
			errorSum += oNeuron.delta * oNeuron.weights[i]
		}
		hNeuron.delta = errorSum * sigmoidDerivative(hNeuron.output)
	}

	// 4. Update Weights & Biases in Parallel
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
		for _, neuron := range layer.neurons {
			go func(n *Neuron, in []float64) {
				defer wg.Done()
				for i := range n.weights {
					n.weights[i] += LearningRate * n.delta * in[i]
				}
				n.bias += LearningRate * n.delta
			}(neuron, prevOutput)
		}
		wg.Wait()
	}
}

func main() {
	fmt.Println("================================================================")
	fmt.Printf("OAI (Ouchi de AI) - Intent Classification DNN Model\n")
	fmt.Printf("URL: %s\n", URL)
	fmt.Printf("Developer Identifier: %s\n", Identifier)
	fmt.Println("Concurrency Mode: Go Goroutines (Parallelized Matrix Math)")
	fmt.Println("================================================================")

	// Categories: [Hobby, Work, Tech, Creative, Life]
	labels := []string{"Hobby", "Work", "Tech", "Creative", "Life"}

	// Simulated training data (User intents encoded as vectors)
	// Example: "How to cook salmon?" -> predominantly Life/Hobby
	trainingData := [][]float64{
		{0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.5}, // Sample 1
		{0.0, 0.0, 0.8, 0.2, 0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0}, // Sample 2
	}
	// Target classifications
	targets := [][]float64{
		{0.8, 0.0, 0.0, 0.0, 0.2}, // Target 1: Hobby focused
		{0.0, 0.0, 0.9, 0.0, 0.1}, // Target 2: Tech focused
	}

	dnn := NewDNN()

	fmt.Printf("Training started on OAI Intent Datasets for %d epochs...\n", Epochs)
	startTime := time.Now()

	for epoch := 1; epoch <= Epochs; epoch++ {
		for i := range trainingData {
			dnn.Train(trainingData[i], targets[i])
		}

		if epoch%200 == 0 {
			totalMse := 0.0
			for i := range trainingData {
				outputs := dnn.Forward(trainingData[i])
				mse := 0.0
				for j := range outputs {
					mse += math.Pow(targets[i][j]-outputs[j], 2)
				}
				totalMse += (mse / float64(OutputSize))
			}
			fmt.Printf("Epoch %d/%d - Average MSE: %.10f\n", epoch, Epochs, totalMse/float64(len(trainingData)))
		}
	}

	duration := time.Since(startTime)
	fmt.Printf("\nTraining completed in %v\n", duration)

	fmt.Println("\nModel Verification (Prediction Test):")
	for i := range trainingData {
		outputs := dnn.Forward(trainingData[i])
		fmt.Printf("Input Sample %d -> Predicted Category: ", i+1)
		
		maxIdx := 0
		maxVal := 0.0
		for j, val := range outputs {
			if val > maxVal {
				maxVal = val
				maxIdx = j
			}
		}
		fmt.Printf("%s (Confidence: %.2f%%)\n", labels[maxIdx], maxVal*100)
	}

	fmt.Printf("\n[SUCCESS] OAI Intent DNN model successfully created and parallelized for %s.\n", Identifier)
}
