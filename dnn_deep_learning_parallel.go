package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// DNN Configuration
const (
	InputSize    = 4
	HiddenSize   = 8
	OutputSize   = 1
	LearningRate = 0.05
	Epochs       = 10000
)

// Activation function: Sigmoid
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of Sigmoid
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Neuron structure
type Neuron struct {
	weights []float64
	bias    float64
	output  float64
	delta   float64
}

// Layer structure
type Layer struct {
	neurons []*Neuron
}

// DNN structure
type DNN struct {
	layers []*Layer
}

// NewDNN initializes the network with random weights
func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	
	// Hidden Layer
	hidden := &Layer{neurons: make([]*Neuron, HiddenSize)}
	for i := 0; i < HiddenSize; i++ {
		n := &Neuron{weights: make([]float64, InputSize), bias: rand.NormFloat64()}
		for j := 0; j < InputSize; j++ {
			n.weights[j] = rand.NormFloat64() * math.Sqrt(2.0/float64(InputSize))
		}
		hidden.neurons[i] = n
	}

	// Output Layer
	output := &Layer{neurons: make([]*Neuron, OutputSize)}
	for i := 0; i < OutputSize; i++ {
		n := &Neuron{weights: make([]float64, HiddenSize), bias: rand.NormFloat64()}
		for j := 0; j < HiddenSize; j++ {
			n.weights[j] = rand.NormFloat64() * math.Sqrt(2.0/float64(HiddenSize))
		}
		output.neurons[i] = n
	}

	return &DNN{layers: []*Layer{hidden, output}}
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

// Backpropagation with Goroutines
func (net *DNN) Train(input, target []float64) {
	// 1. Forward Pass
	net.Forward(input)

	// 2. Output Layer Deltas
	outLayer := net.layers[1]
	for i, neuron := range outLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * sigmoidDerivative(neuron.output)
	}

	// 3. Hidden Layer Deltas
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
	fmt.Println("  Deep Learning (深層学習) - Parallel DNN with Goroutines")
	fmt.Println("  Directory: C:\\Users\\S111478")
	fmt.Println("================================================================")

	// Training data (XOR-like or simple logic)
	// Input: [A, B, C, D] -> Target: (A AND B) OR (C AND D)
	inputs := [][]float64{
		{1, 1, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0},
		{0, 0, 1, 1}, {0, 0, 1, 0}, {0, 0, 0, 1}, {1, 1, 1, 1},
	}
	targets := [][]float64{
		{1}, {0}, {0}, {0},
		{1}, {0}, {0}, {1},
	}

	dnn := NewDNN()

	fmt.Printf("Training started (%d epochs)...\n", Epochs)
	start := time.Now()

	for epoch := 1; epoch <= Epochs; epoch++ {
		for i := range inputs {
			dnn.Train(inputs[i], targets[i])
		}

		if epoch%1000 == 0 {
			err := 0.0
			for i := range inputs {
				out := dnn.Forward(inputs[i])
				err += math.Abs(targets[i][0] - out[0])
			}
			fmt.Printf("Epoch %d: Average Error = %.6f\n", epoch, err/float64(len(inputs)))
		}
	}

	fmt.Printf("\nTraining completed in %v\n", time.Since(start))

	fmt.Println("\nVerification:")
	for i, in := range inputs {
		out := dnn.Forward(in)
		fmt.Printf("Input: %v | Target: %.1f | Predicted: %.6f\n", in, targets[i][0], out[0])
	}

	fmt.Println("\n[Success] DNN Goroutine implementation is ready.")
}
