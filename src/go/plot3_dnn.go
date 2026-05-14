package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// Deep Learning Parameters
const (
	InputSize    = 2    // (x, y) coordinates
	HiddenSize   = 16
	OutputSize   = 1    // Probability of being inside the circle
	LearningRate = 0.05
	Epochs       = 2000
	TrainData    = 1000
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

// NewDNN initializes a new DNN
func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	
	// Hidden Layer
	hidden := &Layer{neurons: make([]*Neuron, HiddenSize)}
	for i := 0; i < HiddenSize; i++ {
		n := &Neuron{weights: make([]float64, InputSize)}
		for j := 0; j < InputSize; j++ {
			n.weights[j] = rand.NormFloat64() * 0.1
		}
		n.bias = rand.NormFloat64() * 0.1
		hidden.neurons[i] = n
	}

	// Output Layer
	output := &Layer{neurons: make([]*Neuron, OutputSize)}
	for i := 0; i < OutputSize; i++ {
		n := &Neuron{weights: make([]float64, HiddenSize)}
		for j := 0; j < HiddenSize; j++ {
			n.weights[j] = rand.NormFloat64() * 0.1
		}
		n.bias = rand.NormFloat64() * 0.1
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

// Train updates weights using backpropagation
func (net *DNN) Train(input, target []float64) {
	// Forward pass
	net.Forward(input)

	// Output Layer Deltas
	outLayer := net.layers[1]
	for i, neuron := range outLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * sigmoidDerivative(neuron.output)
	}

	// Hidden Layer Deltas
	hiddenLayer := net.layers[0]
	for i, hNeuron := range hiddenLayer.neurons {
		var errorSum float64
		for _, oNeuron := range outLayer.neurons {
			errorSum += oNeuron.delta * oNeuron.weights[i]
		}
		hNeuron.delta = errorSum * sigmoidDerivative(hNeuron.output)
	}

	// Update Weights & Biases with Goroutines
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
	fmt.Println("Deep Learning (深層学習) - DNN (ディープニューラルネットワーク)")
	fmt.Println("Goroutine-based Monte Carlo Pi Estimation Simulation")
	fmt.Println("Converting C:\\plot\\plot3.py logic to Go DNN")
	fmt.Println("================================================================")

	// Prepare Training Data
	inputs := make([][]float64, TrainData)
	targets := make([][]float64, TrainData)
	for i := 0; i < TrainData; i++ {
		x := rand.Float64()*2 - 1 // -1 to 1
		y := rand.Float64()*2 - 1 // -1 to 1
		inputs[i] = []float64{x, y}
		if x*x+y*y <= 1.0 {
			targets[i] = []float64{1.0}
		} else {
			targets[i] = []float64{0.0}
		}
	}

	dnn := NewDNN()

	fmt.Printf("Training started for %d epochs...\n", Epochs)
	startTime := time.Now()

	for epoch := 1; epoch <= Epochs; epoch++ {
		// Shuffling data per epoch for better convergence
		rand.Shuffle(len(inputs), func(i, j int) {
			inputs[i], inputs[j] = inputs[j], inputs[i]
			targets[i], targets[j] = targets[j], targets[i]
		})

		for i := 0; i < TrainData; i++ {
			dnn.Train(inputs[i], targets[i])
		}

		if epoch%500 == 0 {
			var totalError float64
			for i := 0; i < 100; i++ {
				out := dnn.Forward(inputs[i])
				totalError += math.Pow(targets[i][0]-out[0], 2)
			}
			fmt.Printf("Epoch %d/%d - Sample MSE: %.6f\n", epoch, Epochs, totalError/100.0)
		}
	}

	duration := time.Since(startTime)
	fmt.Printf("\nTraining completed in %v\n", duration)

	// Estimate Pi using trained DNN
	fmt.Println("\nEstimating Pi using DNN Predictions...")
	testPoints := 10000
	insideCount := 0.0
	for i := 0; i < testPoints; i++ {
		x := rand.Float64()*2 - 1
		y := rand.Float64()*2 - 1
		out := dnn.Forward([]float64{x, y})
		if out[0] >= 0.5 { // Thresholding probability
			insideCount++
		}
	}

	piEstimate := (insideCount / float64(testPoints)) * 4.0
	fmt.Printf("Predicted Points Inside Circle: %.0f / %d\n", insideCount, testPoints)
	fmt.Printf("Estimated Pi: %.6f\n", piEstimate)
	fmt.Printf("Error: %.2f%%\n", math.Abs(math.Pi-piEstimate)/math.Pi*100)

	fmt.Println("\nSuccessfully executed Goroutine-based Deep Learning program.")
}
