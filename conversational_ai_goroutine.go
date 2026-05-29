package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
	"time"
)

// --- Configuration ---
const (
	LearningRate = 0.04
	Epochs       = 10000
	Hidden1      = 48
	Hidden2      = 32
	Hidden3      = 16
)

// --- Data Structures ---

type Intent struct {
	Tag       string
	Patterns  []string
	Responses []string
}

type TrainingData struct {
	Input  []float64
	Target []float64
}

// --- Math ---

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// --- Neural Network ---

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

func NewLayer(size, inputSize int) *Layer {
	l := &Layer{neurons: make([]*Neuron, size)}
	for i := 0; i < size; i++ {
		n := &Neuron{weights: make([]float64, inputSize), bias: rand.NormFloat64()}
		for j := 0; j < inputSize; j++ {
			n.weights[j] = rand.NormFloat64() * math.Sqrt(2.0/float64(inputSize))
		}
		l.neurons[i] = n
	}
	return l
}

func NewDNN(inputSize, h1, h2, h3, outputSize int) *DNN {
	rand.Seed(time.Now().UnixNano())
	return &DNN{
		layers: []*Layer{
			NewLayer(h1, inputSize),
			NewLayer(h2, h1),
			NewLayer(h3, h2),
			NewLayer(outputSize, h3),
		},
	}
}

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

func (net *DNN) Train(input, target []float64) {
	net.Forward(input)

	// Output Layer Deltas
	outLayer := net.layers[len(net.layers)-1]
	for i, neuron := range outLayer.neurons {
		neuron.delta = (target[i] - neuron.output) * sigmoidDerivative(neuron.output)
	}

	// Hidden Layer Deltas
	for l := len(net.layers) - 2; l >= 0; l-- {
		layer := net.layers[l]
		nextLayer := net.layers[l+1]
		for i, neuron := range layer.neurons {
			var errorSum float64
			for _, nextNeuron := range nextLayer.neurons {
				errorSum += nextNeuron.delta * nextNeuron.weights[i]
			}
			neuron.delta = errorSum * sigmoidDerivative(neuron.output)
		}
	}

	// Update Weights & Biases
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

// --- NLP ---

func tokenize(text string) []string {
	text = strings.ToLower(text)
	text = strings.ReplaceAll(text, "?", "")
	text = strings.ReplaceAll(text, "!", "")
	text = strings.ReplaceAll(text, ".", "")
	text = strings.ReplaceAll(text, ",", "")
	return strings.Fields(text)
}

func bagOfWords(tokenizedSentence []string, allWords []string) []float64 {
	bag := make([]float64, len(allWords))
	for _, w := range tokenizedSentence {
		for i, word := range allWords {
			if word == w {
				bag[i] = 1.0
			}
		}
	}
	return bag
}

// --- Execution ---

func main() {
	intents := []Intent{
		{
			Tag: "greeting",
			Patterns: []string{"hi", "hello", "hey", "good morning", "good evening", "howdy"},
			Responses: []string{"Hello! How can I assist you with high-precision simulations today?", "Greetings. System status: Optimal.", "Hi there! Ready for some DNN training?"},
		},
		{
			Tag: "goodbye",
			Patterns: []string{"bye", "goodbye", "exit", "quit", "stop", "see you"},
			Responses: []string{"Terminating session. Goodbye!", "Shutting down neural link. Take care.", "Simulation ended. See you next time!"},
		},
		{
			Tag: "precision",
			Patterns: []string{"precision", "long double", "80-bit", "accuracy", "rounding error", "floating point"},
			Responses: []string{"We use 'long double' to minimize rounding errors in chaotic systems.", "Accuracy is paramount. Our C simulations utilize the full 80-bit precision on win32.", "Numerical stability is achieved through high-precision floating point math."},
		},
		{
			Tag: "goroutine",
			Patterns: []string{"goroutine", "parallel", "concurrency", "multi-thread", "performance", "speed"},
			Responses: []string{"Goroutines allow us to parallelize neuron calculations across all CPU cores.", "We use sync.WaitGroup to coordinate the parallel forward and backward passes.", "Performance is boosted by distributing matrix operations via Go's concurrency model."},
		},
		{
			Tag: "chaos",
			Patterns: []string{"chaos", "lorenz", "attractor", "rk4", "runge-kutta", "butterfly effect"},
			Responses: []string{"The Lorenz system demonstrates how small changes lead to chaotic results.", "Our RK4 (Runge-Kutta 4th order) implementation ensures stable integration of chaotic paths.", "Chaos theory is a key research area in this high-precision workspace."},
		},
		{
			Tag: "finance",
			Patterns: []string{"finance", "bep", "break even", "profit", "margin", "cost"},
			Responses: []string{"The Break-Even Point (BEP) is calculated using our precision-based DNN models.", "We analyze contribution margins and safety ratios for business profitability.", "Financial modeling here combines traditional accounting with deep learning predictions."},
		},
		{
			Tag: "rag",
			Patterns: []string{"rag", "retrieval", "knowledge base", "vector", "cosine similarity"},
			Responses: []string{"Our RAG system uses cosine similarity to retrieve the most relevant scientific data.", "The knowledge base in rag_data.json powers our context-aware responses.", "We implement vector-based retrieval in C for maximum efficiency."},
		},
	}

	// Prepare Vocabulary
	allWords := []string{}
	wordMap := make(map[string]bool)
	for _, intent := range intents {
		for _, pattern := range intent.Patterns {
			tokens := tokenize(pattern)
			for _, token := range tokens {
				if !wordMap[token] {
					allWords = append(allWords, token)
					wordMap[token] = true
				}
			}
		}
	}

	// Prepare Training Data
	training := []TrainingData{}
	for i, intent := range intents {
		for _, pattern := range intent.Patterns {
			input := bagOfWords(tokenize(pattern), allWords)
			target := make([]float64, len(intents))
			target[i] = 1.0
			training = append(training, TrainingData{Input: input, Target: target})
		}
	}

	// Train
	fmt.Println("--- Conversational AI Construction (DNN + Goroutines) ---")
	fmt.Printf("Vocabulary Size: %d | Intents: %d\n", len(allWords), len(intents))
	dnn := NewDNN(len(allWords), Hidden1, Hidden2, Hidden3, len(intents))

	start := time.Now()
	for epoch := 1; epoch <= Epochs; epoch++ {
		for _, data := range training {
			dnn.Train(data.Input, data.Target)
		}
		if epoch%2000 == 0 {
			fmt.Printf("Progress: %d/%d epochs...\n", epoch, Epochs)
		}
	}
	fmt.Printf("Construction complete in %v\n", time.Since(start))

	// Interactive
	fmt.Println("\n--- AI Ready for Dialogue (type 'quit' to exit) ---")
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()
		if strings.ToLower(input) == "quit" {
			break
		}

		prediction := dnn.Forward(bagOfWords(tokenize(input), allWords))
		maxIdx := 0
		maxVal := -1.0
		for i, val := range prediction {
			if val > maxVal {
				maxVal = val
				maxIdx = i
			}
		}

		if maxVal > 0.45 {
			intent := intents[maxIdx]
			fmt.Printf("AI: %s (Confidence: %.2f%%)\n", intent.Responses[rand.Intn(len(intent.Responses))], maxVal*100)
		} else {
			fmt.Println("AI: I need more data to classify that intent accurately.")
		}
	}
}
