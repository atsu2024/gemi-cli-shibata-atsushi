package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// Deep Learning (深層学習) & DNN (ディープニューラルネットワーク) Configuration
const (
	InputSize    = 10
	HiddenSize   = 512
	OutputSize   = 1
	LearningRate = 0.05
	Epochs       = 200
)

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of sigmoid function
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Neuron represents a single neuron
type Neuron struct {
	weights []float64
	bias    float64
	output  float64
	delta   float64
}

// Layer represents a layer
type Layer struct {
	neurons []*Neuron
}

// DNN represents the network
type DNN struct {
	layers []*Layer
}

// NewDNN initializes the network
func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	net := &DNN{}

	// Hidden Layer
	hiddenLayer := &Layer{}
	for i := 0; i < HiddenSize; i++ {
		neuron := &Neuron{
			weights: make([]float64, InputSize),
			bias:    rand.Float64()*2 - 1,
		}
		for j := 0; j < InputSize; j++ {
			neuron.weights[j] = rand.Float64()*2 - 1
		}
		hiddenLayer.neurons = append(hiddenLayer.neurons, neuron)
	}
	net.layers = append(net.layers, hiddenLayer)

	// Output Layer
	outputLayer := &Layer{}
	for i := 0; i < OutputSize; i++ {
		neuron := &Neuron{
			weights: make([]float64, HiddenSize),
			bias:    rand.Float64()*2 - 1,
		}
		for j := 0; j < HiddenSize; j++ {
			neuron.weights[j] = rand.Float64()*2 - 1
		}
		outputLayer.neurons = append(outputLayer.neurons, neuron)
	}
	net.layers = append(net.layers, outputLayer)

	return net
}

// Forward propagation
func (net *DNN) Forward(input []float64) []float64 {
	currentInput := input
	for _, layer := range net.layers {
		nextInput := make([]float64, len(layer.neurons))
		var wg sync.WaitGroup
		for i, neuron := range layer.neurons {
			wg.Add(1)
			go func(idx int, n *Neuron, in []float64) {
				defer wg.Done()
				sum := n.bias
				for j, val := range in {
					sum += val * n.weights[j]
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

// Train the network
func (net *DNN) Train(input, target []float64) {
	net.Forward(input)

	// Output layer
	outLayer := net.layers[len(net.layers)-1]
	for i, neuron := range outLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * sigmoidDerivative(neuron.output)
	}

	// Hidden layer
	layer := net.layers[0]
	nextLayer := net.layers[1]
	for i, neuron := range layer.neurons {
		var errorSum float64
		for _, nextNeuron := range nextLayer.neurons {
			errorSum += nextNeuron.delta * nextNeuron.weights[i]
		}
		neuron.delta = errorSum * sigmoidDerivative(neuron.output)
	}

	// Update weights
	prevInput := input
	for _, layer := range net.layers {
		for _, neuron := range layer.neurons {
			for j, val := range prevInput {
				neuron.weights[j] += LearningRate * neuron.delta * val
			}
			neuron.bias += LearningRate * neuron.delta
		}
		nextPrevInput := make([]float64, len(layer.neurons))
		for i, n := range layer.neurons {
			nextPrevInput[i] = n.output
		}
		prevInput = nextPrevInput
	}
}

func main() {
	fmt.Println("DNN Implementation - fix1.go")
	dnn := NewDNN()

	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	// Pad inputs to InputSize
	paddedInputs := make([][]float64, 4)
	for i := range inputs {
		paddedInputs[i] = make([]float64, InputSize)
		copy(paddedInputs[i], inputs[i])
	}
	targets := [][]float64{{0}, {1}, {1}, {0}}

	fmt.Printf("Training for %d epochs...\n", Epochs)
	for e := 1; e <= Epochs; e++ {
		for i := 0; i < 4; i++ {
			dnn.Train(paddedInputs[i], targets[i])
		}
	}

	fmt.Println("Testing XOR:")
	for i := 0; i < 4; i++ {
		pred := dnn.Forward(paddedInputs[i])
		fmt.Printf("In: %v -> Target: %v | Pred: %.4f\n", inputs[i], targets[i], pred[0])
	}
}
