package main

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Deep Learning Parameters
const (
	Res          = 64
	InputSize    = Res * Res
	Hidden1Size  = 512
	Hidden2Size  = 128
	Hidden3Size  = 512
	OutputSize   = InputSize
	LearningRate = 0.02
	Epochs       = 500
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
	// Xavier Initialization
	limit := math.Sqrt(6.0 / float64(inDim+outDim))
	for i := range l.weights {
		l.weights[i] = make([]float64, outDim)
		for j := range l.weights[i] {
			l.weights[i][j] = (rand.Float64()*2 - 1) * limit
		}
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
			NewLayer(Hidden2Size, Hidden3Size),
			NewLayer(Hidden3Size, OutputSize),
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
	path := filepath.Join("data", "shibata_atsushi01.png")
	fmt.Printf("Deep Learning: Loading %s\n", path)
	
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		fmt.Printf("Decode error: %v\n", err)
		return
	}

	// Resize and Grayscale
	inputData := make([]float64, InputSize)
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	for y := 0; y < Res; y++ {
		for x := 0; x < Res; x++ {
			ix, iy := x*w/Res, y*h/Res
			r, g, b, _ := img.At(ix, iy).RGBA()
			inputData[y*Res+x] = (0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 65535.0
		}
	}

	dnn := NewDNN()
	fmt.Printf("Starting Deep Learning (4-Layer DNN) with Goroutines...\n")
	start := time.Now()
	for epoch := 1; epoch <= Epochs; epoch++ {
		dnn.Forward(inputData)
		dnn.Backprop(inputData, inputData)
		if epoch%50 == 0 || epoch == 1 {
			mse := 0.0
			out := dnn.layers[len(dnn.layers)-1].outputs
			for i := range inputData {
				mse += math.Pow(inputData[i]-out[i], 2)
			}
			fmt.Printf("Epoch %d/%d, MSE: %.8f\n", epoch, Epochs, mse/InputSize)
		}
	}
	fmt.Printf("Deep Learning completed in %v\n", time.Since(start))

	// Save Output
	outImg := image.NewGray(image.Rect(0, 0, Res, Res))
	finalOut := dnn.layers[len(dnn.layers)-1].outputs
	for i, val := range finalOut {
		outImg.SetGray(i%Res, i/Res, color.Gray{Y: uint8(val * 255)})
	}
	
	outPath := "deep_shibata_output.png"
	f, _ := os.Create(outPath)
	defer f.Close()
	png.Encode(f, outImg)
	fmt.Printf("DNN image saved to %s\n", outPath)
}
