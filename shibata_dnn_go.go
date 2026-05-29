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
	HiddenSize1  = 20
	HiddenSize2  = 10
	OutputSize   = 1
	LearningRate = 0.005
	Epochs       = 5000
)

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of sigmoid
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

// NewDNN initializes a multi-layer neural network
func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	
	sizes := []int{InputSize, HiddenSize1, HiddenSize2, OutputSize}
	layers := make([]*Layer, len(sizes)-1)

	for i := 0; i < len(sizes)-1; i++ {
		layer := &Layer{neurons: make([]*Neuron, sizes[i+1])}
		for j := 0; j < sizes[i+1]; j++ {
			n := &Neuron{weights: make([]float64, sizes[i]), bias: rand.NormFloat64()}
			// Xavier/Glorot initialization for weights
			stdDev := math.Sqrt(2.0 / float64(sizes[i]+sizes[i+1]))
			for k := 0; k < sizes[i]; k++ {
				n.weights[k] = rand.NormFloat64() * stdDev
			}
			layer.neurons[j] = n
		}
		layers[i] = layer
	}

	return &DNN{layers: layers}
}

// Forward pass with Goroutines for parallel neuron calculation
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

// Train performs one iteration of backpropagation
func (net *DNN) Train(input, target []float64) {
	// 1. Forward Pass
	net.Forward(input)

	// 2. Output Layer Error
	outLayer := net.layers[len(net.layers)-1]
	for i, neuron := range outLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * sigmoidDerivative(neuron.output)
	}

	// 3. Backpropagate Deltas through Hidden Layers
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

	// 4. Update Weights and Biases (Parallel)
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
	fmt.Println("========================================================================")
	fmt.Println("  [Deep Learning] 深層学習 - ディープニューラルネットワーク (DNN)")
	fmt.Println("  Go言語 (Golang) + Goroutine 並列計算実装")
	fmt.Println("========================================================================")

	// Sample Training Data
	numSamples := 20
	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		inputs[i] = make([]float64, InputSize)
		avg := 0.0
		for j := 0; j < InputSize; j++ {
			inputs[i][j] = rand.Float64()
			avg += inputs[i][j]
		}
		avg /= float64(InputSize)
		// Target is 1 if average > 0.5, else 0 (Classification)
		if avg > 0.5 {
			targets[i] = []float64{1.0}
		} else {
			targets[i] = []float64{0.0}
		}
	}

	dnn := NewDNN()

	fmt.Printf("Training: %d Epochs, Learning Rate: %.4f\n", Epochs, LearningRate)
	start := time.Now()

	for epoch := 1; epoch <= Epochs; epoch++ {
		for i := range inputs {
			dnn.Train(inputs[i], targets[i])
		}

		if epoch%500 == 0 {
			mse := 0.0
			for i := range inputs {
				pred := dnn.Forward(inputs[i])
				mse += math.Pow(targets[i][0]-pred[0], 2)
			}
			fmt.Printf("Epoch %d/%d - MSE: %.6f\n", epoch, Epochs, mse/float64(numSamples))
		}
	}

	fmt.Printf("Training completed in %v\n", time.Since(start))

	fmt.Println("\n--- Validation (Test Prediction) ---")
	testInput := make([]float64, InputSize)
	for i := 0; i < InputSize; i++ {
		testInput[i] = rand.Float64()
	}
	prediction := dnn.Forward(testInput)
	fmt.Printf("Input Features Avg: %.4f\n", func(arr []float64) float64 {
		s := 0.0
		for _, v := range arr {
			s += v
		}
		return s / float64(len(arr))
	}(testInput))
	fmt.Printf("DNN Prediction: %.6f\n", prediction[0])

	fmt.Println("\nGO言語プログラムのビルドと実行に成功しました。")
}
