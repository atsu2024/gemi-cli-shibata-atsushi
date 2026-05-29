package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// 深層学習 (Deep Learning) / ディープニューラルネットワーク (DNN)
// Goroutineを使用した並列処理実装

const (
	InputSize    = 8
	HiddenSize   = 16
	OutputSize   = 4
	LearningRate = 0.05
	Epochs       = 1000
)

// 活性化関数: Tanh
func tanh(x float64) float64 {
	return math.Tanh(x)
}

// Tanhの微分
func tanhDerivative(x float64) float64 {
	return 1.0 - math.Pow(x, 2)
}

type Neuron struct {
	weights []float64
	bias    float64
	output  float64
	delta   float64
}

type Layer struct {
	neurons []*Neuron
}

type DNN struct {
	layers []*Layer
}

func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	
	// 入力層 -> 隠れ層
	hidden := &Layer{neurons: make([]*Neuron, HiddenSize)}
	for i := 0; i < HiddenSize; i++ {
		n := &Neuron{weights: make([]float64, InputSize)}
		for j := 0; j < InputSize; j++ {
			n.weights[j] = rand.Float64()*2 - 1 // -1.0 to 1.0
		}
		hidden.neurons[i] = n
	}

	// 隠れ層 -> 出力層
	output := &Layer{neurons: make([]*Neuron, OutputSize)}
	for i := 0; i < OutputSize; i++ {
		n := &Neuron{weights: make([]float64, HiddenSize)}
		for j := 0; j < HiddenSize; j++ {
			n.weights[j] = rand.Float64()*2 - 1
		}
		output.neurons[i] = n
	}

	return &DNN{layers: []*Layer{hidden, output}}
}

// Goroutineを使用した並列順伝播
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
				n.output = tanh(sum)
				nextInput[idx] = n.output
			}(i, neuron, currentInput)
		}
		wg.Wait()
		currentInput = nextInput
	}
	return currentInput
}

// 並列重み更新を含む学習プロセス
func (net *DNN) Train(input, target []float64) {
	// 1. 順伝播
	net.Forward(input)

	// 2. 出力層の誤差計算
	outLayer := net.layers[len(net.layers)-1]
	for i, neuron := range outLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * tanhDerivative(neuron.output)
	}

	// 3. 隠れ層の誤差計算
	hiddenLayer := net.layers[0]
	for i, hNeuron := range hiddenLayer.neurons {
		var errorSum float64
		for _, oNeuron := range outLayer.neurons {
			errorSum += oNeuron.delta * oNeuron.weights[i]
		}
		hNeuron.delta = errorSum * tanhDerivative(hNeuron.output)
	}

	// 4. 重みとバイアスの更新 (並列化)
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
	fmt.Println("  Deep Neural Network (DNN) with Goroutines in Go")
	fmt.Println("  深層学習 (Deep Learning) - 並列処理実装")
	fmt.Println("================================================================")

	dnn := NewDNN()

	// サンプルデータ作成
	inputData := make([]float64, InputSize)
	for i := range inputData {
		inputData[i] = rand.Float64()
	}
	targetData := make([]float64, OutputSize)
	for i := range targetData {
		targetData[i] = rand.Float64()*2 - 1
	}

	fmt.Printf("Training started for %d epochs...\n", Epochs)
	startTime := time.Now()

	for epoch := 1; epoch <= Epochs; epoch++ {
		dnn.Train(inputData, targetData)

		if epoch%200 == 0 {
			outputs := dnn.Forward(inputData)
			errorTotal := 0.0
			for i := range outputs {
				errorTotal += math.Abs(targetData[i] - outputs[i])
			}
			fmt.Printf("Epoch %d/%d - Avg Error: %.6f\n", epoch, Epochs, errorTotal/float64(OutputSize))
		}
	}

	fmt.Printf("\nTraining completed in %v\n", time.Since(startTime))

	fmt.Println("\nResult Verification:")
	finalOutputs := dnn.Forward(inputData)
	for i := 0; i < OutputSize; i++ {
		fmt.Printf("Target: %8.4f | Output: %8.4f | Diff: %8.4f\n", targetData[i], finalOutputs[i], math.Abs(targetData[i]-finalOutputs[i]))
	}
	fmt.Println("================================================================")
}
