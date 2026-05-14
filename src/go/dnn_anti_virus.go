package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Activation Functions
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * x
}

func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1.0
	}
	return 0.01
}

func sigmoid(x float64) float64 {
	if x > 100 {
		return 1.0
	}
	if x < -100 {
		return 0.0
	}
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// DeepMLP structure
type DeepMLP struct {
	numLayers  int
	layerSizes []int
	nodes      [][]float64
	weights    [][][]float64
	biases     [][]float64
	deltas     [][]float64

	// Adam Optimizer buffers
	mWeights [][][]float64
	vWeights [][][]float64
	mBiases  [][]float64
	vBiases  [][]float64
	beta1    float64
	beta2    float64
	epsilon  float64
	t        int
}

func NewDeepMLP(layerSizes []int) *DeepMLP {
	numLayers := len(layerSizes)
	mlp := &DeepMLP{
		numLayers:  numLayers,
		layerSizes: layerSizes,
		nodes:      make([][]float64, numLayers),
		biases:     make([][]float64, numLayers),
		deltas:     make([][]float64, numLayers),
		weights:    make([][][]float64, numLayers-1),
		mWeights:   make([][][]float64, numLayers-1),
		vWeights:   make([][][]float64, numLayers-1),
		mBiases:    make([][]float64, numLayers),
		vBiases:    make([][]float64, numLayers),
		beta1:      0.9,
		beta2:      0.999,
		epsilon:    1e-8,
		t:          0,
	}

	for i := 0; i < numLayers; i++ {
		mlp.nodes[i] = make([]float64, layerSizes[i])
		mlp.deltas[i] = make([]float64, layerSizes[i])
		if i > 0 {
			mlp.biases[i] = make([]float64, layerSizes[i])
			mlp.mBiases[i] = make([]float64, layerSizes[i])
			mlp.vBiases[i] = make([]float64, layerSizes[i])

			mlp.weights[i-1] = make([][]float64, layerSizes[i-1])
			mlp.mWeights[i-1] = make([][]float64, layerSizes[i-1])
			mlp.vWeights[i-1] = make([][]float64, layerSizes[i-1])

			fanIn := layerSizes[i-1]
			stddev := math.Sqrt(2.0 / float64(fanIn))

			for j := 0; j < layerSizes[i-1]; j++ {
				mlp.weights[i-1][j] = make([]float64, layerSizes[i])
				mlp.mWeights[i-1][j] = make([]float64, layerSizes[i])
				mlp.vWeights[i-1][j] = make([]float64, layerSizes[i])
				for k := 0; k < layerSizes[i]; k++ {
					mlp.weights[i-1][j][k] = rand.NormFloat64() * stddev
				}
			}
		}
	}
	return mlp
}

func (mlp *DeepMLP) ForwardProp(inputs []float64) {
	for i := 0; i < mlp.layerSizes[0]; i++ {
		mlp.nodes[0][i] = inputs[i]
	}

	for i := 1; i < mlp.numLayers; i++ {
		var wg sync.WaitGroup
		wg.Add(mlp.layerSizes[i])
		for j := 0; j < mlp.layerSizes[i]; j++ {
			go func(layerIdx, nodeIdx int) {
				defer wg.Done()
				activation := mlp.biases[layerIdx][nodeIdx]
				for k := 0; k < mlp.layerSizes[layerIdx-1]; k++ {
					activation += mlp.nodes[layerIdx-1][k] * mlp.weights[layerIdx-1][k][nodeIdx]
				}

				if layerIdx == mlp.numLayers-1 {
					mlp.nodes[layerIdx][nodeIdx] = sigmoid(activation)
				} else {
					mlp.nodes[layerIdx][nodeIdx] = relu(activation)
				}
			}(i, j)
		}
		wg.Wait()
	}
}

func (mlp *DeepMLP) BackPropAdam(targets []float64, lr float64) {
	last := mlp.numLayers - 1
	mlp.t++

	for i := 0; i < mlp.layerSizes[last]; i++ {
		errorVal := targets[i] - mlp.nodes[last][i]
		mlp.deltas[last][i] = errorVal * sigmoidDerivative(mlp.nodes[last][i])
	}

	for i := last - 1; i > 0; i-- {
		var wg sync.WaitGroup
		wg.Add(mlp.layerSizes[i])
		for j := 0; j < mlp.layerSizes[i]; j++ {
			go func(layerIdx, nodeIdx int) {
				defer wg.Done()
				var errorVal float64
				for k := 0; k < mlp.layerSizes[layerIdx+1]; k++ {
					errorVal += mlp.deltas[layerIdx+1][k] * mlp.weights[layerIdx][nodeIdx][k]
				}
				mlp.deltas[layerIdx][nodeIdx] = errorVal * reluDerivative(mlp.nodes[layerIdx][nodeIdx])
			}(i, j)
		}
		wg.Wait()
	}

	for i := 1; i < mlp.numLayers; i++ {
		var wg sync.WaitGroup
		wg.Add(mlp.layerSizes[i])
		for j := 0; j < mlp.layerSizes[i]; j++ {
			go func(layerIdx, nodeIdx int) {
				defer wg.Done()
				gb := mlp.deltas[layerIdx][nodeIdx]
				mlp.mBiases[layerIdx][nodeIdx] = mlp.beta1*mlp.mBiases[layerIdx][nodeIdx] + (1.0-mlp.beta1)*gb
				mlp.vBiases[layerIdx][nodeIdx] = mlp.beta2*mlp.vBiases[layerIdx][nodeIdx] + (1.0-mlp.beta2)*gb*gb
				mHat := mlp.mBiases[layerIdx][nodeIdx] / (1.0 - math.Pow(mlp.beta1, float64(mlp.t)))
				vHat := mlp.vBiases[layerIdx][nodeIdx] / (1.0 - math.Pow(mlp.beta2, float64(mlp.t)))
				mlp.biases[layerIdx][nodeIdx] += lr * mHat / (math.Sqrt(vHat) + mlp.epsilon)

				for k := 0; k < mlp.layerSizes[layerIdx-1]; k++ {
					gw := mlp.deltas[layerIdx][nodeIdx] * mlp.nodes[layerIdx-1][k]
					mlp.mWeights[layerIdx-1][k][nodeIdx] = mlp.beta1*mlp.mWeights[layerIdx-1][k][nodeIdx] + (1.0-mlp.beta1)*gw
					mlp.vWeights[layerIdx-1][k][nodeIdx] = mlp.beta2*mlp.vWeights[layerIdx-1][k][nodeIdx] + (1.0-mlp.beta2)*gw*gw
					mwHat := mlp.mWeights[layerIdx-1][k][nodeIdx] / (1.0 - math.Pow(mlp.beta1, float64(mlp.t)))
					vwHat := mlp.vWeights[layerIdx-1][k][nodeIdx] / (1.0 - math.Pow(mlp.beta2, float64(mlp.t)))
					mlp.weights[layerIdx-1][k][nodeIdx] += lr * mwHat / (math.Sqrt(vwHat) + mlp.epsilon)
				}
			}(i, j)
		}
		wg.Wait()
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())
	layers := []int{2, 16, 32, 16, 8, 1}
	mlp := NewDeepMLP(layers)

	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	targets := [][]float64{{0}, {1}, {1}, {0}}

	fmt.Println("Starting Go-Powered DNN Training (Adam + Goroutines)...")
	start := time.Now()
	for epoch := 0; epoch < 2000; epoch++ {
		for i := 0; i < 4; i++ {
			mlp.ForwardProp(inputs[i])
			mlp.BackPropAdam(targets[i], 0.001)
		}
		if epoch%200 == 0 {
			mse := 0.0
			for i := 0; i < 4; i++ {
				mlp.ForwardProp(inputs[i])
				mse += math.Pow(targets[i][0]-mlp.nodes[len(layers)-1][0], 2)
			}
			fmt.Printf("Epoch %d: MSE = %.15f\n", epoch, mse/4.0)
		}
	}
	fmt.Printf("Training completed in %v\n", time.Since(start))

	fmt.Println("\n--- Interactive Prediction Mode ---")
	fmt.Println("Enter two numbers (0 or 1) separated by space (e.g., '0 1'). Type 'exit' to quit.")
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		line := scanner.Text()
		if strings.ToLower(line) == "exit" {
			break
		}
		parts := strings.Fields(line)
		if len(parts) == 2 {
			in1, _ := strconv.ParseFloat(parts[0], 64)
			in2, _ := strconv.ParseFloat(parts[1], 64)
			mlp.ForwardProp([]float64{in1, in2})
			pred := mlp.nodes[len(layers)-1][0]
			class := 0
			if pred > 0.5 {
				class = 1
			}
			fmt.Printf("Prediction: %.15f (Class: %d)\n", pred, class)
		}
	}
}
