package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// Global Configuration for High-Precision Deep Learning
const (
	InputSize    = 10
	Hidden1Size  = 64
	Hidden2Size  = 32
	OutputSize   = 10
	LearningRate = 0.01
	Epochs       = 1000
)

// Sigmoid activation function with high precision
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of sigmoid for backpropagation
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Neuron structure for high-precision modeling
type Neuron struct {
	weights []float64
	bias    float64
	output  float64
	delta   float64
}

// Layer structure to handle parallel neuron processing
type Layer struct {
	neurons []*Neuron
}

// DNN structure representing a Deep Neural Network
type DNN struct {
	layers []*Layer
}

// NewDNN initializes the network with optimized random weights (Xavier/He initialization style)
func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	
	sizes := []int{InputSize, Hidden1Size, Hidden2Size, OutputSize}
	net := &DNN{layers: make([]*Layer, len(sizes)-1)}

	for i := 0; i < len(sizes)-1; i++ {
		layer := &Layer{neurons: make([]*Neuron, sizes[i+1])}
		for j := 0; j < sizes[i+1]; j++ {
			n := &Neuron{
				weights: make([]float64, sizes[i]),
				bias:    rand.NormFloat64() * 0.1,
			}
			// Xavier Initialization
			stdDev := math.Sqrt(2.0 / float64(sizes[i]))
			for k := 0; k < sizes[i]; k++ {
				n.weights[k] = rand.NormFloat64() * stdDev
			}
			layer.neurons[j] = n
		}
		net.layers[i] = layer
	}
	return net
}

// Forward pass using Goroutines for massive parallelism across neurons
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

// Train performs parallel backpropagation and weight updates
func (net *DNN) Train(input, target []float64) {
	// 1. Forward Pass
	net.Forward(input)

	// 2. Output Layer Deltas
	outputLayer := net.layers[len(net.layers)-1]
	for i, neuron := range outputLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * sigmoidDerivative(neuron.output)
	}

	// 3. Backpropagate Deltas through Hidden Layers
	for l := len(net.layers) - 2; l >= 0; l-- {
		layer := net.layers[l]
		nextLayer := net.layers[l+1]
		
		var wg sync.WaitGroup
		wg.Add(len(layer.neurons))
		
		for i, neuron := range layer.neurons {
			go func(idx int, n *Neuron) {
				defer wg.Done()
				var errorSum float64
				for _, nextNeuron := range nextLayer.neurons {
					errorSum += nextNeuron.delta * nextNeuron.weights[idx]
				}
				n.delta = errorSum * sigmoidDerivative(n.output)
			}(i, neuron)
		}
		wg.Wait()
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
	fmt.Println("  Deep Learning (深層学習) - Custom High-Precision DNN")
	fmt.Println("  Parallel Architecture: Go Goroutines (並列処理)")
	fmt.Println("  Location: C:\\Users\\S111478")
	fmt.Println("================================================================")

	// Generate synthetic training data (Identity mapping / Autoencoder task)
	dataSize := 20
	inputs := make([][]float64, dataSize)
	targets := make([][]float64, dataSize)
	for i := 0; i < dataSize; i++ {
		inputs[i] = make([]float64, InputSize)
		for j := 0; j < InputSize; j++ {
			inputs[i][j] = rand.Float64()
		}
		targets[i] = inputs[i] // Task: Learn to reconstruct the input
	}

	dnn := NewDNN()

	fmt.Printf("Starting High-Precision Training (%d epochs)...\n", Epochs)
	start := time.Now()

	for epoch := 1; epoch <= Epochs; epoch++ {
		for i := range inputs {
			dnn.Train(inputs[i], targets[i])
		}

		if epoch%100 == 0 {
			mse := 0.0
			for i := range inputs {
				out := dnn.Forward(inputs[i])
				for j := range out {
					mse += math.Pow(targets[i][j]-out[j], 2)
				}
			}
			fmt.Printf("Epoch %d: Mean Squared Error = %.10f\n", epoch, mse/float64(dataSize*InputSize))
		}
	}

	fmt.Printf("\nTraining completed in %v\n", time.Since(start))

	fmt.Println("\nVerification of Learning (Final Sample):")
	testInput := inputs[0]
	predicted := dnn.Forward(testInput)
	for i := 0; i < 5; i++ {
		fmt.Printf("Index %d | Target: %.6f | Predicted: %.6f | Diff: %.6f\n", 
			i, testInput[i], predicted[i], math.Abs(testInput[i]-predicted[i]))
	}

	fmt.Println("\n[Success] dnn_shibata_precision.go has been created, built, and executed.")
}
