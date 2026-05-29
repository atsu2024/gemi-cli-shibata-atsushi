package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

// -----------------------------------------------------------------------------
// Deep Learning (深層学習) & DNN (ディープニューラルネットワーク)
// Parallel Execution using Go Goroutines
// Inspired by: https://application-developm-5fdq.bolt.host
// -----------------------------------------------------------------------------

const (
	InputSize    = 3  // e.g., Fixed Cost, Variable Cost per unit, Price per unit
	HiddenSize   = 64
	OutputSize   = 1  // Profit
	LearningRate = 0.01
	Epochs       = 2000
	ParallelMin  = 16 // Minimum neurons to trigger parallelism
)

// Activation function: Tanh (common for DNNs)
func tanh(x float64) float64 {
	return math.Tanh(x)
}

// Derivative of Tanh
func tanhDerivative(x float64) float64 {
	return 1.0 - x*x
}

type Neuron struct {
	weights []float64
	bias    float64
	output  float64
	delta   float64
}

type Layer struct {
	neurons []*Neuron
	outputs []float64
}

type DNN struct {
	layers []*Layer
}

func NewLayer(size int, prevSize int) *Layer {
	layer := &Layer{
		neurons: make([]*Neuron, size),
		outputs: make([]float64, size),
	}
	for i := 0; i < size; i++ {
		n := &Neuron{weights: make([]float64, prevSize)}
		// Xavier/Glorot initialization
		limit := math.Sqrt(6.0 / float64(size+prevSize))
		for j := 0; j < prevSize; j++ {
			n.weights[j] = rand.Float64()*2*limit - limit
		}
		layer.neurons[i] = n
	}
	return layer
}

func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	hidden := NewLayer(HiddenSize, InputSize)
	output := NewLayer(OutputSize, HiddenSize)
	return &DNN{layers: []*Layer{hidden, output}}
}

// Forward Pass with Goroutines
func (net *DNN) Forward(input []float64) []float64 {
	currentInput := input
	numCPU := runtime.NumCPU()

	for _, layer := range net.layers {
		numNeurons := len(layer.neurons)
		
		var wg sync.WaitGroup
		if numNeurons >= ParallelMin {
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
						n.output = tanh(sum)
						layer.outputs[i] = n.output
					}
				}(start, end, currentInput)
			}
			wg.Wait()
		} else {
			for i, n := range layer.neurons {
				sum := n.bias
				for k, w := range n.weights {
					sum += w * currentInput[k]
				}
				n.output = tanh(sum)
				layer.outputs[i] = n.output
			}
		}
		currentInput = layer.outputs
	}
	return currentInput
}

// Backpropagation with Goroutines
func (net *DNN) Train(input, target []float64) {
	net.Forward(input)

	// 1. Calculate Deltas
	// Output Layer
	outLayer := net.layers[len(net.layers)-1]
	for i, n := range outLayer.neurons {
		n.delta = (target[i] - n.output) * tanhDerivative(n.output)
	}

	// Hidden Layer(s)
	for lIdx := len(net.layers) - 2; lIdx >= 0; lIdx-- {
		layer := net.layers[lIdx]
		nextLayer := net.layers[lIdx+1]
		
		for i, n := range layer.neurons {
			var errorSum float64
			for _, nextN := range nextLayer.neurons {
				errorSum += nextN.delta * nextN.weights[i]
			}
			n.delta = errorSum * tanhDerivative(n.output)
		}
	}

	// 2. Update Weights
	numCPU := runtime.NumCPU()
	for lIdx, layer := range net.layers {
		var prevOut []float64
		if lIdx == 0 {
			prevOut = input
		} else {
			prevOut = net.layers[lIdx-1].outputs
		}

		numNeurons := len(layer.neurons)
		if numNeurons >= ParallelMin {
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
				}(start, end, prevOut)
			}
			wg.Wait()
		} else {
			for _, n := range layer.neurons {
				for k := range n.weights {
					n.weights[k] += LearningRate * n.delta * prevOut[k]
				}
				n.bias += LearningRate * n.delta
			}
		}
	}
}

// Generate Break-even Point Training Data
// Input: [FixedCost, VarCostPerUnit, PricePerUnit]
// Output: [ProfitAt100Units] (Normalized)
func generateData() ([][]float64, [][]float64) {
	inputs := [][]float64{}
	targets := [][]float64{}
	for i := 0; i < 1000; i++ {
		fc := rand.Float64()*1000 + 500  // 500-1500
		vc := rand.Float64()*50 + 10    // 10-60
		pr := rand.Float64()*100 + 70   // 70-170
		
		inputs = append(inputs, []float64{fc / 1500, vc / 60, pr / 170})
		
		// Profit at 100 units: 100*Price - (FixedCost + 100*VarCost)
		profit := 100*pr - (fc + 100*vc)
		targets = append(targets, []float64{math.Tanh(profit / 5000)}) // Normalize to -1..1
	}
	return inputs, targets
}

func main() {
	fmt.Println("================================================================")
	fmt.Println("DNN (Deep Neural Network) with Go Goroutines")
	fmt.Println("Scenario: Break-even Analysis (損益分岐点分析)")
	fmt.Println("Inspired by: Bolt.new Application Logic")
	fmt.Println("================================================================")

	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Printf("Running on %d CPUs\n", runtime.NumCPU())

	inputs, targets := generateData()
	dnn := NewDNN()

	fmt.Printf("Training on %d samples for %d epochs...\n", len(inputs), Epochs)
	start := time.Now()

	for epoch := 1; epoch <= Epochs; epoch++ {
		for i := range inputs {
			dnn.Train(inputs[i], targets[i])
		}
		if epoch%200 == 0 {
			// Calculate total loss
			var totalLoss float64
			for i := range inputs {
				out := dnn.Forward(inputs[i])
				totalLoss += math.Pow(targets[i][0]-out[0], 2)
			}
			fmt.Printf("Epoch %d/%d - Loss: %.6f\n", epoch, Epochs, totalLoss/float64(len(inputs)))
		}
	}

	duration := time.Since(start)
	fmt.Printf("\nTraining completed in %v\n", duration)

	fmt.Println("\n--- Test Results (Break-even Point Estimation) ---")
	testCases := [][]float64{
		{1000, 30, 100}, // Profit @ 100: 100*100 - (1000 + 100*30) = 10000 - 4000 = 6000
		{1500, 60, 70},  // Profit @ 100: 100*70 - (1500 + 100*60) = 7000 - 7500 = -500
		{800, 20, 150},  // Profit @ 100: 100*150 - (800 + 100*20) = 15000 - 2800 = 12200
	}

	for _, tc := range testCases {
		normIn := []float64{tc[0] / 1500, tc[1] / 60, tc[2] / 170}
		out := dnn.Forward(normIn)
		
		// De-normalize profit (approximate)
		// We used Tanh(profit / 5000), so atanh(out)*5000
		estimatedProfit := math.Atanh(out[0]) * 5000
		
		actualProfit := 100*tc[2] - (tc[0] + 100*tc[1])
		
		fmt.Printf("FC: %.0f, VC: %.0f, Price: %.0f | Actual: %.0f | DNN Predict: %.0f\n", 
			tc[0], tc[1], tc[2], actualProfit, estimatedProfit)
	}

	fmt.Println("\nBuilding and execution successful.")
}
