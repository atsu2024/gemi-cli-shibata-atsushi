package main

import (
	"fmt"
	"math"
	"math/rand"
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

// For regression, we use identity for the last layer
func identity(x float64) float64 {
	return x
}

func identityDerivative(x float64) float64 {
	return 1.0
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
					mlp.nodes[layerIdx][nodeIdx] = identity(activation)
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
		mlp.deltas[last][i] = errorVal * identityDerivative(mlp.nodes[last][i])
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

	// MathDraw1.py equivalents: y = ax + b and y = ax^2 + b
	// We will train a DNN to learn both. Input will be [x, type_flag]
	// type_flag: 0 for linear, 1 for parabola
	
	layers := []int{2, 32, 64, 32, 1}
	mlp := NewDeepMLP(layers)

	// Generate training data
	const numSamples = 1000
	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)

	a_lin, b_lin := 2.0, 5.0      // y = 2x + 5
	a_par, b_par := 0.5, -10.0    // y = 0.5x^2 - 10

	for i := 0; i < numSamples; i++ {
		x := (rand.Float64() * 20.0) - 10.0 // x in [-10, 10]
		if i%2 == 0 {
			// Linear
			inputs[i] = []float64{x, 0.0}
			targets[i] = []float64{a_lin*x + b_lin}
		} else {
			// Parabola
			inputs[i] = []float64{x, 1.0}
			targets[i] = []float64{a_par*x*x + b_par}
		}
	}

	fmt.Println("Starting Go-Powered DNN Math Learning (Adam + Goroutines)...")
	fmt.Printf("Learning Linear: y = %.1fx + %.1f\n", a_lin, b_lin)
	fmt.Printf("Learning Parabola: y = %.1fxx + %.1f\n", a_par, b_par)
	
	start := time.Now()
	epochs := 500
	lr := 0.001
	
	for epoch := 0; epoch <= epochs; epoch++ {
		// Shuffle indices for stochastic gradient descent
		idx := rand.Perm(numSamples)
		for _, i := range idx {
			mlp.ForwardProp(inputs[i])
			mlp.BackPropAdam(targets[i], lr)
		}

		if epoch%(epochs/10) == 0 {
			mse := 0.0
			for i := 0; i < numSamples; i++ {
				mlp.ForwardProp(inputs[i])
				mse += math.Pow(targets[i][0]-mlp.nodes[len(layers)-1][0], 2)
			}
			fmt.Printf("Epoch %d/%d: MSE = %.6f\n", epoch, epochs, mse/float64(numSamples))
		}
	}
	fmt.Printf("Training completed in %v\n", time.Since(start))

	fmt.Println("\n--- Testing Trained DNN Predictions ---")
	testPoints := []float64{-5, 0, 5}
	for _, x := range testPoints {
		// Test Linear
		mlp.ForwardProp([]float64{x, 0.0})
		predLin := mlp.nodes[len(layers)-1][0]
		actualLin := a_lin*x + b_lin
		
		// Test Parabola
		mlp.ForwardProp([]float64{x, 1.0})
		predPar := mlp.nodes[len(layers)-1][0]
		actualPar := a_par*x*x + b_par

		fmt.Printf("x = %5.1f | Linear   - Pred: %8.4f, Actual: %8.4f, Error: %8.4f\n", x, predLin, actualLin, math.Abs(predLin-actualLin))
		fmt.Printf("          | Parabola - Pred: %8.4f, Actual: %8.4f, Error: %8.4f\n", predPar, actualPar, math.Abs(predPar-actualPar))
	}
}
