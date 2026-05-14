package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
)

// Hyperparameters
const (
	InputNodes  = 2
	HiddenNodes = 4
	OutputNodes = 1
	LearningRate = 0.1
	Epochs       = 10000
)

// Activation functions
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// NeuralNetwork structure
type NeuralNetwork struct {
	wih [][]float64 // Weights: Input to Hidden
	who [][]float64 // Weights: Hidden to Output
	bh  []float64   // Bias: Hidden
	bo  []float64   // Bias: Output
}

func NewNeuralNetwork() *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())
	nn := &NeuralNetwork{
		wih: make([][]float64, InputNodes),
		who: make([][]float64, HiddenNodes),
		bh:  make([]float64, HiddenNodes),
		bo:  make([]float64, OutputNodes),
	}

	for i := range nn.wih {
		nn.wih[i] = make([]float64, HiddenNodes)
		for j := range nn.wih[i] {
			nn.wih[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(InputNodes))
		}
	}

	for i := range nn.who {
		nn.who[i] = make([]float64, OutputNodes)
		for j := range nn.who[i] {
			nn.who[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(HiddenNodes))
		}
	}

	for i := range nn.bh {
		nn.bh[i] = 0.0
	}
	for i := range nn.bo {
		nn.bo[i] = 0.0
	}

	return nn
}

func (nn *NeuralNetwork) Train(inputs []float64, targets []float64) {
	// --- Forward Pass with Goroutines ---
	hiddenOutputs := make([]float64, HiddenNodes)
	var wg sync.WaitGroup

	wg.Add(HiddenNodes)
	for j := 0; j < HiddenNodes; j++ {
		go func(hIdx int) {
			defer wg.Done()
			sum := nn.bh[hIdx]
			for i := 0; i < InputNodes; i++ {
				sum += inputs[i] * nn.wih[i][hIdx]
			}
			hiddenOutputs[hIdx] = sigmoid(sum)
		}(j)
	}
	wg.Wait()

	finalOutputs := make([]float64, OutputNodes)
	wg.Add(OutputNodes)
	for k := 0; k < OutputNodes; k++ {
		go func(oIdx int) {
			defer wg.Done()
			sum := nn.bo[oIdx]
			for j := 0; j < HiddenNodes; j++ {
				sum += hiddenOutputs[j] * nn.who[j][oIdx]
			}
			finalOutputs[oIdx] = sigmoid(sum)
		}(k)
	}
	wg.Wait()

	// --- Backpropagation ---
	
	// Output Layer Errors
	outputErrors := make([]float64, OutputNodes)
	for k := 0; k < OutputNodes; k++ {
		outputErrors[k] = targets[k] - finalOutputs[k]
	}

	// Output Layer Gradients
	outputGradients := make([]float64, OutputNodes)
	for k := 0; k < OutputNodes; k++ {
		outputGradients[k] = outputErrors[k] * sigmoidDerivative(finalOutputs[k]) * LearningRate
	}

	// Hidden Layer Errors
	hiddenErrors := make([]float64, HiddenNodes)
	wg.Add(HiddenNodes)
	for j := 0; j < HiddenNodes; j++ {
		go func(hIdx int) {
			defer wg.Done()
			var err float64
			for k := 0; k < OutputNodes; k++ {
				err += outputErrors[k] * nn.who[hIdx][k]
			}
			hiddenErrors[hIdx] = err
		}(j)
	}
	wg.Wait()

	// Hidden Layer Gradients
	hiddenGradients := make([]float64, HiddenNodes)
	for j := 0; j < HiddenNodes; j++ {
		hiddenGradients[j] = hiddenErrors[j] * sigmoidDerivative(hiddenOutputs[j]) * LearningRate
	}

	// --- Update Weights and Biases with Goroutines ---

	// Update WHO and BO
	wg.Add(HiddenNodes)
	for j := 0; j < HiddenNodes; j++ {
		go func(hIdx int) {
			defer wg.Done()
			for k := 0; k < OutputNodes; k++ {
				nn.who[hIdx][k] += outputGradients[k] * hiddenOutputs[hIdx]
			}
		}(j)
	}
	for k := 0; k < OutputNodes; k++ {
		nn.bo[k] += outputGradients[k]
	}
	wg.Wait()

	// Update WIH and BH
	wg.Add(InputNodes)
	for i := 0; i < InputNodes; i++ {
		go func(iIdx int) {
			defer wg.Done()
			for j := 0; j < HiddenNodes; j++ {
				nn.wih[iIdx][j] += hiddenGradients[j] * inputs[iIdx]
			}
		}(i)
	}
	for j := 0; j < HiddenNodes; j++ {
		nn.bh[j] += hiddenGradients[j]
	}
	wg.Wait()
}

func (nn *NeuralNetwork) Predict(inputs []float64) []float64 {
	// Simplified forward pass (synchronous for prediction)
	hiddenOutputs := make([]float64, HiddenNodes)
	for j := 0; j < HiddenNodes; j++ {
		sum := nn.bh[j]
		for i := 0; i < InputNodes; i++ {
			sum += inputs[i] * nn.wih[i][j]
		}
		hiddenOutputs[j] = sigmoid(sum)
	}

	finalOutputs := make([]float64, OutputNodes)
	for k := 0; k < OutputNodes; k++ {
		sum := nn.bo[k]
		for j := 0; j < HiddenNodes; j++ {
			sum += hiddenOutputs[j] * nn.who[j][k]
		}
		finalOutputs[k] = sigmoid(sum)
	}
	return finalOutputs
}

func main() {
	fmt.Println("Techgym DNN (Deep Learning) Starting...")
	fmt.Println("Loading XOR data from xor_data.csv...")

	file, err := os.Open("xor_data.csv")
	if err != nil {
		fmt.Printf("Error opening file: %v\n", err)
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Printf("Error reading CSV: %v\n", err)
		return
	}

	inputs := make([][]float64, len(records))
	targets := make([][]float64, len(records))

	for i, record := range records {
		inputs[i] = make([]float64, 2)
		targets[i] = make([]float64, 1)
		
		val1, _ := strconv.ParseFloat(record[0], 64)
		val2, _ := strconv.ParseFloat(record[1], 64)
		target, _ := strconv.ParseFloat(record[2], 64)
		
		inputs[i][0] = val1
		inputs[i][1] = val2
		targets[i][0] = target
	}

	nn := NewNeuralNetwork()

	fmt.Printf("Training DNN with Goroutines for %d epochs...\n", Epochs)
	start := time.Now()
	for epoch := 0; epoch < Epochs; epoch++ {
		// Shuffle data would be better, but keeping it simple
		for i := range inputs {
			nn.Train(inputs[i], targets[i])
		}
		if epoch%1000 == 0 {
			fmt.Printf("Epoch %d completed\n", epoch)
		}
	}
	fmt.Printf("Training finished in %v\n", time.Since(start))

	fmt.Println("\nTesting Predictions:")
	for i := range inputs {
		prediction := nn.Predict(inputs[i])
		fmt.Printf("Input: %v, Target: %v, Prediction: %.4f\n", inputs[i], targets[i], prediction[0])
	}
}
