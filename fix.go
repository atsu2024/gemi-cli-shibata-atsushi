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
	HiddenSize   = 128
	OutputSize   = 1
	LearningRate = 0.1
	Epochs       = 100
)

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of sigmoid function
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Neuron represents a single neuron in the network
type Neuron struct {
	weights []float64
	bias    float64
	output  float64
	delta   float64
}

// Layer represents a layer in the neural network
type Layer struct {
	neurons []*Neuron
}

// DNN represents the deep neural network
type DNN struct {
	layers []*Layer
}

// NewDNN creates and initializes a new DNN
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

// Forward performs forward propagation in parallel
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

// Train performs backpropagation and weight updates
func (net *DNN) Train(input, target []float64) {
	net.Forward(input)

	// Output layer
	outLayer := net.layers[len(net.layers)-1]
	for i, neuron := range outLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * sigmoidDerivative(neuron.output)
	}

	// Hidden layers
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

	// Update weights
	prevInput := input
	for _, layer := range net.layers {
		for _, neuron := range layer.neurons {
			for j, val := range prevInput {
				neuron.weights[j] += LearningRate * neuron.delta * val
			}
			neuron.bias += LearningRate * neuron.delta
		}
		// next layer's prevInput is this layer's output
		nextPrevInput := make([]float64, len(layer.neurons))
		for i, n := range layer.neurons {
			nextPrevInput[i] = n.output
		}
		prevInput = nextPrevInput
	}
}

func main() {
	fmt.Println("================================================================")
	fmt.Println("  Deep Learning (深層学習) - DNN (ディープニューラルネットワーク)")
	fmt.Println("  High-Performance Implementation using Go Goroutines")
	fmt.Println("================================================================")

	dnn := NewDNN()

	// Sample Training Data (XOR-like but for InputSize)
	inputs := make([][]float64, 4)
	targets := make([][]float64, 4)
	for i := 0; i < 4; i++ {
		inputs[i] = make([]float64, InputSize)
		for j := 0; j < InputSize; j++ {
			inputs[i][j] = rand.Float64()
		}
		targets[i] = []float64{rand.Float64()}
	}

	fmt.Printf("Training started for %d epochs using Goroutines...\n", Epochs)
	startTime := time.Now()

	for e := 1; e <= Epochs; e++ {
		for i := 0; i < len(inputs); i++ {
			dnn.Train(inputs[i], targets[i])
		}
		if e%10 == 0 {
			fmt.Printf("Epoch %d/%d completed\n", e, Epochs)
		}
	}

	fmt.Printf("Training finished in %v\n", time.Since(startTime))

	fmt.Println("\nTesting predictions:")
	for i := 0; i < 3; i++ {
		prediction := dnn.Forward(inputs[i])
		fmt.Printf("Data [%d] - Target: %.6f | Predicted: %.6f\n", i, targets[i][0], prediction[0])
	}

	fmt.Println("\nGO言語プログラム (fix.go) のビルドと実行が正常に完了しました。")
}
