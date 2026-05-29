package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

// Deep Learning (深層学習) & DNN (ディープニューラルネットワーク) Configuration
// SECV Model Configuration
const (
	InputSize     = 8
	Hidden1Size   = 16
	Hidden2Size   = 16
	OutputSize    = 1
	LearningRate  = 0.05
	Epochs        = 1000
	WorkerCount   = 4
)

// Sigmoid activation function (シグモイド関数)
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of sigmoid for backpropagation
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
	outputs []float64
}

// NewLayer initializes a layer with random weights
func NewLayer(size int, inputSize int) *Layer {
	layer := &Layer{
		neurons: make([]*Neuron, size),
		outputs: make([]float64, size),
	}
	for i := 0; i < size; i++ {
		n := &Neuron{
			weights: make([]float64, inputSize),
			bias:    rand.NormFloat64() * 0.1,
		}
		for j := 0; j < inputSize; j++ {
			// Xavier/Glorot initialization
			n.weights[j] = rand.NormFloat64() * math.Sqrt(2.0/float64(inputSize+size))
		}
		layer.neurons[i] = n
	}
	return layer
}

// DNN represents the SECV Deep Neural Network structure
type DNN struct {
	layers []*Layer
}

// NewDNN initializes the SECV DNN with hidden layers
func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	
	l1 := NewLayer(Hidden1Size, InputSize)
	l2 := NewLayer(Hidden2Size, Hidden1Size)
	l3 := NewLayer(OutputSize, Hidden2Size)

	return &DNN{layers: []*Layer{l1, l2, l3}}
}

// Forward pass using Goroutines for parallel neuron processing
func (net *DNN) Forward(input []float64) []float64 {
	currentInput := input
	for _, layer := range net.layers {
		var wg sync.WaitGroup
		wg.Add(len(layer.neurons))

		for i, neuron := range layer.neurons {
			go func(idx int, n *Neuron, in []float64, l *Layer) {
				defer wg.Done()
				sum := n.bias
				for k, w := range n.weights {
					sum += w * in[k]
				}
				n.output = sigmoid(sum)
				l.outputs[idx] = n.output
			}(i, neuron, currentInput, layer)
		}
		wg.Wait()
		currentInput = layer.outputs
	}
	return currentInput
}

// Train performs backpropagation and weight updates in parallel
func (net *DNN) Train(input, target []float64) {
	// 1. Forward Pass
	net.Forward(input)

	// 2. Backpropagation - Calculate Deltas
	// Output Layer
	outLayer := net.layers[len(net.layers)-1]
	for i, neuron := range outLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * sigmoidDerivative(neuron.output)
	}

	// Hidden Layers (in reverse)
	for l := len(net.layers) - 2; l >= 0; l-- {
		layer := net.layers[l]
		nextLayer := net.layers[l+1]
		for i, neuron := range layer.neurons {
			var errorSum float64
			for _, nextNeuron := range nextLayer.neurons {
				errorSum += nextNeuron.delta * nextNeuron.weights[i]
			}
			neuron.delta = errorSum * sigmoidDerivative(neuron.output)
		}
	}

	// 3. Update Weights & Biases in Parallel
	for lIdx, layer := range net.layers {
		var prevOutput []float64
		if lIdx == 0 {
			prevOutput = input
		} else {
			prevOutput = net.layers[lIdx-1].outputs
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
	runtime.GOMAXPROCS(runtime.NumCPU())

	fmt.Println("================================================================")
	fmt.Println("  Deep Learning (深層学習) - DNN (ディープニューラルネットワーク)")
	fmt.Println("  Project SECV - High-Performance Parallel Go Implementation")
	fmt.Println("================================================================")

	// Generate synthetic training data
	// Let's train the network to predict if the sum of inputs is > threshold
	trainCount := 100
	inputs := make([][]float64, trainCount)
	targets := make([][]float64, trainCount)
	for i := 0; i < trainCount; i++ {
		inputs[i] = make([]float64, InputSize)
		sum := 0.0
		for j := 0; j < InputSize; j++ {
			inputs[i][j] = rand.Float64()
			sum += inputs[i][j]
		}
		if sum > float64(InputSize)/2.0 {
			targets[i] = []float64{1.0}
		} else {
			targets[i] = []float64{0.0}
		}
	}

	dnn := NewDNN()

	fmt.Printf("Training SECV Model with %d layers and Goroutines...\n", len(dnn.layers))
	fmt.Printf("Epochs: %d, Learning Rate: %.4f\n", Epochs, LearningRate)
	
	startTime := time.Now()

	for epoch := 1; epoch <= Epochs; epoch++ {
		for i := range inputs {
			dnn.Train(inputs[i], targets[i])
		}

		if epoch%100 == 0 {
			mse := 0.0
			for i := range inputs {
				out := dnn.Forward(inputs[i])
				mse += math.Pow(targets[i][0]-out[0], 2)
			}
			fmt.Printf("Epoch %d/%d - Mean Squared Error: %.8f\n", epoch, Epochs, mse/float64(trainCount))
		}
	}

	duration := time.Since(startTime)
	fmt.Printf("\nTraining completed in %v\n", duration)

	fmt.Println("\nValidation of SECV Model Predictions:")
	for i := 0; i < 5; i++ {
		testInput := make([]float64, InputSize)
		sum := 0.0
		for j := 0; j < InputSize; j++ {
			testInput[j] = rand.Float64()
			sum += testInput[j]
		}
		target := 0.0
		if sum > float64(InputSize)/2.0 {
			target = 1.0
		}
		
		prediction := dnn.Forward(testInput)
		fmt.Printf("Test [%d] - Sum: %.2f | Target: %.1f | Prediction: %.4f\n", i, sum, target, prediction[0])
	}

	fmt.Println("\n[SUCCESS] SECV DNN Goroutine Go program built and executed successfully.")
}
