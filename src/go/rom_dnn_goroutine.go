package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"
)

const (
	InputSize    = 16
	HiddenSize   = 64
	OutputSize   = 16
	LearningRate = 0.05
	Epochs       = 100
	RomFilePath  = `C:\Users\S111478\Box\個人_S111478\2022\rom.bin`
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

// Forward pass using Goroutines
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

// Train using Goroutines
func (net *DNN) Train(input, target []float64) {
	net.Forward(input)

	// Calculate Output Layer Deltas
	outLayer := net.layers[1]
	for i, neuron := range outLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * sigmoidDerivative(neuron.output)
	}

	// Calculate Hidden Layer Deltas
	hiddenLayer := net.layers[0]
	for i, hNeuron := range hiddenLayer.neurons {
		var errorSum float64
		for _, oNeuron := range outLayer.neurons {
			errorSum += oNeuron.delta * oNeuron.weights[i]
		}
		hNeuron.delta = errorSum * sigmoidDerivative(hNeuron.output)
	}

	// Update Weights & Biases in Parallel
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

func readRomData(filePath string) ([]float64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	stats, err := file.Stat()
	if err != nil {
		return nil, err
	}

	size := stats.Size()
	bytes := make([]byte, size)
	_, err = io.ReadFull(file, bytes)
	if err != nil {
		return nil, err
	}

	data := make([]float64, size)
	for i, b := range bytes {
		data[i] = float64(b) / 255.0
	}
	return data, nil
}

func main() {
	fmt.Println("================================================================")
	fmt.Println("DNN training with ROM data using Go Goroutines")
	fmt.Println("Input ROM File:", RomFilePath)
	fmt.Println("================================================================")

	romData, err := readRomData(RomFilePath)
	if err != nil {
		fmt.Printf("Error reading ROM data: %v\n", err)
		return
	}

	if len(romData) < InputSize+OutputSize {
		fmt.Println("Not enough data in ROM file for training.")
		return
	}

	fmt.Printf("Read %d float64 values from ROM.\n", len(romData))

	// Prepare training samples (InputSize chunks)
	var samples [][]float64
	for i := 0; i+InputSize+OutputSize <= len(romData); i += InputSize {
		samples = append(samples, romData[i:i+InputSize])
	}

	dnn := NewDNN()

	fmt.Printf("Training started for %d epochs using %d samples...\n", Epochs, len(samples))
	startTime := time.Now()

	for epoch := 1; epoch <= Epochs; epoch++ {
		totalMse := 0.0
		for _, sample := range samples {
			// Autoencoder style: target is same as input
			dnn.Train(sample, sample)
			
			if epoch%10 == 0 {
				outputs := dnn.Forward(sample)
				for i := range outputs {
					totalMse += math.Pow(sample[i]-outputs[i], 2)
				}
			}
		}

		if epoch%10 == 0 {
			fmt.Printf("Epoch %d/%d - Average MSE: %.10f\n", epoch, Epochs, totalMse/float64(len(samples)*InputSize))
		}
	}

	duration := time.Since(startTime)
	fmt.Printf("\nTraining completed in %v\n", duration)

	fmt.Println("\nVerification on the first sample:")
	testSample := samples[0]
	outputs := dnn.Forward(testSample)
	for i := 0; i < 8 && i < len(testSample); i++ {
		fmt.Printf("ROM Data: %12.8f -> DNN Output: %12.8f\n", testSample[i], outputs[i])
	}

	fmt.Println("\nSuccessfully executed the ROM-based Deep Learning program with Goroutines.")
}
