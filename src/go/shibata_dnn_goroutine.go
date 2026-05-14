package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// Deep Learning (深層学習) Parameters
// Identifier: shibata2017meister@gmail.com
const (
	InputSize    = 10
	HiddenSize   = 32
	OutputSize   = 10
	LearningRate = 0.1
	Epochs       = 500
	Identifier   = "shibata2017meister@gmail.com"
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
	fmt.Println("----------------------------------------------------------------")
	fmt.Printf("Deep Learning (深層学習) - DNN (ディープニューラルネットワーク)\n")
	fmt.Printf("User Identifier: %s\n", Identifier)
	fmt.Println("Parallel Execution using Go Goroutines")
	fmt.Println("----------------------------------------------------------------")

	// Simulated data
	inputData := make([]float64, InputSize)
	for i := range inputData {
		inputData[i] = rand.Float64()
	}
	targetData := inputData // Simple autoencoder task

	dnn := NewDNN()

	fmt.Printf("Training started for %d epochs...\n", Epochs)
	startTime := time.Now()

	for epoch := 1; epoch <= Epochs; epoch++ {
		dnn.Train(inputData, targetData)

		if epoch%50 == 0 {
			outputs := dnn.Forward(inputData)
			mse := 0.0
			for i := range outputs {
				mse += math.Pow(targetData[i]-outputs[i], 2)
			}
			fmt.Printf("Epoch %d/%d - MSE: %.10f\n", epoch, Epochs, mse/float64(InputSize))
		}
	}

	duration := time.Since(startTime)
	fmt.Printf("\nTraining completed in %v\n", duration)

	fmt.Println("\nFinal Output Verification:")
	finalOutputs := dnn.Forward(inputData)
	for i := 0; i < 5; i++ {
		fmt.Printf("Input: %.4f -> DNN Output: %.4f\n", inputData[i], finalOutputs[i])
	}

	fmt.Printf("\nSuccessfully built and executed the Goroutine-based Deep Learning program for %s.\n", Identifier)
}
