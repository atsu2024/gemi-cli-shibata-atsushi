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
	InputSize    = 10
	Hidden1Size  = 64
	Hidden2Size  = 32
	OutputSize   = 10
	LearningRate = 0.05
	Epochs       = 1000
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

type Layer struct {
	weights [][]float64
	biases  []float64
	outputs []float64
	deltas  []float64
}

func NewLayer(inDim, outDim int) *Layer {
	l := &Layer{
		weights: make([][]float64, inDim),
		biases:  make([]float64, outDim),
		outputs: make([]float64, outDim),
		deltas:  make([]float64, outDim),
	}
	// Xavier/Glorot Initialization
	limit := math.Sqrt(6.0 / float64(inDim+outDim))
	for i := range l.weights {
		l.weights[i] = make([]float64, outDim)
		for j := range l.weights[i] {
			l.weights[i][j] = (rand.Float64()*2 - 1) * limit
		}
	}
	for i := range l.biases {
		l.biases[i] = 0
	}
	return l
}

type DNN struct {
	layers []*Layer
}

func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	return &DNN{
		layers: []*Layer{
			NewLayer(InputSize, Hidden1Size),
			NewLayer(Hidden1Size, Hidden2Size),
			NewLayer(Hidden2Size, OutputSize),
		},
	}
}

func (net *DNN) Forward(input []float64) {
	currentInput := input
	for _, l := range net.layers {
		var wg sync.WaitGroup
		wg.Add(len(l.outputs))
		for j := 0; j < len(l.outputs); j++ {
			go func(idx int, in []float64, layer *Layer) {
				defer wg.Done()
				sum := layer.biases[idx]
				for i := 0; i < len(in); i++ {
					sum += in[i] * layer.weights[i][idx]
				}
				layer.outputs[idx] = sigmoid(sum)
			}(j, currentInput, l)
		}
		wg.Wait()
		currentInput = l.outputs
	}
}

func (net *DNN) Backprop(input, target []float64) {
	// Output Layer Deltas
	last := net.layers[len(net.layers)-1]
	for i := 0; i < len(last.outputs); i++ {
		last.deltas[i] = (target[i] - last.outputs[i]) * sigmoidDerivative(last.outputs[i])
	}

	// Hidden Layer Deltas
	for i := len(net.layers) - 2; i >= 0; i-- {
		l := net.layers[i]
		next := net.layers[i+1]
		var wg sync.WaitGroup
		wg.Add(len(l.outputs))
		for j := 0; j < len(l.outputs); j++ {
			go func(idx int) {
				defer wg.Done()
				var err float64
				for k := 0; k < len(next.deltas); k++ {
					err += next.deltas[k] * next.weights[idx][k]
				}
				l.deltas[idx] = err * sigmoidDerivative(l.outputs[idx])
			}(j)
		}
		wg.Wait()
	}

	// Update Weights & Biases
	currentIn := input
	for _, l := range net.layers {
		var wg sync.WaitGroup
		wg.Add(len(l.weights))
		for i := 0; i < len(l.weights); i++ {
			go func(idx int) {
				defer wg.Done()
				for j := 0; j < len(l.outputs); j++ {
					l.weights[idx][j] += LearningRate * l.deltas[j] * currentIn[idx]
				}
			}(i)
		}
		for j := 0; j < len(l.outputs); j++ {
			l.biases[j] += LearningRate * l.deltas[j]
		}
		wg.Wait()
		currentIn = l.outputs
	}
}

func main() {
	fmt.Println("====================================================")
	fmt.Println("Deep Learning (深層学習) & DNN (ディープニューラルネットワーク)")
	fmt.Println("Subject: Conversion from LLM to Deep Learning Focus")
	fmt.Println("Source: llm構築（free版）.pdf concepts")
	fmt.Println("====================================================")

	// Synthetic data for Autoencoder task
	data := make([]float64, InputSize)
	for i := range data {
		data[i] = rand.Float64()
	}

	dnn := NewDNN()
	fmt.Printf("Starting DNN Training with Goroutines (Epochs: %d)...\n", Epochs)
	
	start := time.Now()
	for epoch := 1; epoch <= Epochs; epoch++ {
		dnn.Forward(data)
		dnn.Backprop(data, data) // Training as an autoencoder
		
		if epoch%200 == 0 || epoch == 1 {
			mse := 0.0
			out := dnn.layers[len(dnn.layers)-1].outputs
			for i := range data {
				mse += math.Pow(data[i]-out[i], 2)
			}
			fmt.Printf("Epoch %d/%d, MSE: %.8f\n", epoch, Epochs, mse/InputSize)
		}
	}
	
	duration := time.Since(start)
	fmt.Printf("\nDeep Learning Training Completed in %v\n", duration)
	
	fmt.Println("\nVerification of Output (First 5 values):")
	finalOut := dnn.layers[len(dnn.layers)-1].outputs
	for i := 0; i < 5; i++ {
		fmt.Printf("Input: %.4f -> Output: %.4f\n", data[i], finalOut[i])
	}

	fmt.Println("\nSuccessfully integrated Deep Learning (深層学習) logic using Goroutines.")
}
