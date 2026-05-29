package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

// Deep Learning (深層学習) Parameters
const (
	InputSize    = 10
	HiddenSize   = 32
	OutputSize   = 10
	LearningRate = 0.1
	Epochs       = 500
	ParallelThreshold = 256 // Run sequentially if neurons < this
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
	outputs []float64 // Pre-allocated output buffer
}

// DNN represents the Deep Neural Network structure
type DNN struct {
	layers []*Layer
}

// NewDNN initializes a new DNN with random weights and pre-allocated buffers
func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	
	// Create Hidden Layer
	hidden := &Layer{
		neurons: make([]*Neuron, HiddenSize),
		outputs: make([]float64, HiddenSize),
	}
	for i := 0; i < HiddenSize; i++ {
		n := &Neuron{weights: make([]float64, InputSize)}
		for j := 0; j < InputSize; j++ {
			n.weights[j] = rand.NormFloat64() * math.Sqrt(2.0/float64(InputSize))
		}
		hidden.neurons[i] = n
	}

	// Create Output Layer
	output := &Layer{
		neurons: make([]*Neuron, OutputSize),
		outputs: make([]float64, OutputSize),
	}
	for i := 0; i < OutputSize; i++ {
		n := &Neuron{weights: make([]float64, HiddenSize)}
		for j := 0; j < HiddenSize; j++ {
			n.weights[j] = rand.NormFloat64() * math.Sqrt(2.0/float64(HiddenSize))
		}
		output.neurons[i] = n
	}

	return &DNN{layers: []*Layer{hidden, output}}
}

// Forward pass using Chunked Goroutines or Sequential execution
func (net *DNN) Forward(input []float64) []float64 {
	currentInput := input
	numCPU := runtime.NumCPU()

	for _, layer := range net.layers {
		numNeurons := len(layer.neurons)

		if numNeurons < ParallelThreshold {
			// Sequential execution for small layers
			for i, neuron := range layer.neurons {
				sum := neuron.bias
				for k, w := range neuron.weights {
					sum += w * currentInput[k]
				}
				neuron.output = sigmoid(sum)
				layer.outputs[i] = neuron.output
			}
		} else {
			// Chunked Parallel execution for large layers
			var wg sync.WaitGroup
			chunkSize := (numNeurons + numCPU - 1) / numCPU
			for c := 0; c < numCPU; c++ {
				start := c * chunkSize
				if start >= numNeurons {
					break
				}
				end := start + chunkSize
				if end > numNeurons {
					end = numNeurons
				}

				wg.Add(1)
				go func(s, e int, in []float64) {
					defer wg.Done()
					for i := s; i < e; i++ {
						n := layer.neurons[i]
						sum := n.bias
						for k, w := range n.weights {
							sum += w * in[k]
						}
						n.output = sigmoid(sum)
						layer.outputs[i] = n.output
					}
				}(start, end, currentInput)
			}
			wg.Wait()
		}
		currentInput = layer.outputs
	}
	return currentInput
}

// Backpropagate error and update weights
func (net *DNN) Train(input, target []float64) {
	// 1. Forward Pass
	net.Forward(input)

	// 2. Calculate Output Layer Deltas (Sequential)
	outLayer := net.layers[1]
	for i, neuron := range outLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * sigmoidDerivative(neuron.output)
	}

	// 3. Calculate Hidden Layer Deltas (Sequential)
	hiddenLayer := net.layers[0]
	for i, hNeuron := range hiddenLayer.neurons {
		var errorSum float64
		for _, oNeuron := range outLayer.neurons {
			errorSum += oNeuron.delta * oNeuron.weights[i]
		}
		hNeuron.delta = errorSum * sigmoidDerivative(hNeuron.output)
	}

	// 4. Update Weights & Biases with Chunking/Sequential logic
	numCPU := runtime.NumCPU()
	for lIdx, layer := range net.layers {
		var prevOutput []float64
		if lIdx == 0 {
			prevOutput = input
		} else {
			prevOutput = net.layers[lIdx-1].outputs
		}

		numNeurons := len(layer.neurons)
		if numNeurons < ParallelThreshold {
			for _, neuron := range layer.neurons {
				for i := range neuron.weights {
					neuron.weights[i] += LearningRate * neuron.delta * prevOutput[i]
				}
				neuron.bias += LearningRate * neuron.delta
			}
		} else {
			var wg sync.WaitGroup
			chunkSize := (numNeurons + numCPU - 1) / numCPU
			for c := 0; c < numCPU; c++ {
				start := c * chunkSize
				if start >= numNeurons {
					break
				}
				end := start + chunkSize
				if end > numNeurons {
					end = numNeurons
				}

				wg.Add(1)
				go func(s, e int, in []float64) {
					defer wg.Done()
					for i := s; i < e; i++ {
						n := layer.neurons[i]
						for k := range n.weights {
							n.weights[k] += LearningRate * n.delta * in[k]
						}
						n.bias += LearningRate * n.delta
					}
				}(start, end, prevOutput)
			}
			wg.Wait()
		}
	}
}

func main() {
	fmt.Println("----------------------------------------------------------------")
	fmt.Println("Deep Learning (深層学習) - DNN (ディープニューラルネットワーク)")
	fmt.Println("Parallel Execution using Go Goroutines")
	fmt.Println("Ref: C:\\Users\\S111478\\OneDrive - 三機工業株式会社\\ドキュメント\\gemini llm構築.pdf")
	fmt.Println("----------------------------------------------------------------")

	// Simulated data (could be from C:\Users\S111478\Pictures\image)
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

	fmt.Println("\nSuccessfully built and executed the Goroutine-based Deep Learning program.")
}
