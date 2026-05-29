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

// --- DNN Configuration ---
const (
	LearningRate = 0.05
	Epochs       = 8000
	HiddenSize1  = 32
	HiddenSize2  = 24
	HiddenSize3  = 16
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

// --- Math Functions ---

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// --- Neural Network Components ---

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

	// Update Weights
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

// --- NLP Utils ---

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

// --- Main Chatbot Logic ---

func main() {
	intents := []Intent{
		{
			Tag:       "greeting",
			Patterns:  []string{"hi", "hello", "hey", "good day", "greetings", "morning", "evening", "what's up"},
			Responses: []string{"Hello!", "Hi there!", "How can I help you today?", "Greetings!", "Always at your service!"},
		},
		{
			Tag:       "goodbye",
			Patterns:  []string{"bye", "goodbye", "see you later", "catch you later", "exit", "quit", "stop"},
			Responses: []string{"Goodbye!", "See you soon!", "Have a nice day!", "Take care!", "Shutting down the neural link..."},
		},
		{
			Tag:       "status",
			Patterns:  []string{"how are you", "how is it going", "how are things", "status", "health check"},
			Responses: []string{"I'm doing great, thank you!", "Everything is running smoothly.", "I'm a high-precision AI, always ready!", "System diagnostics: All layers optimal."},
		},
		{
			Tag:       "science",
			Patterns:  []string{"tell me about science", "scientific simulation", "physics", "biot savart", "lorenz system", "rk4", "magnetic field", "numerical", "runge kutta"},
			Responses: []string{"This project focuses on high-precision scientific computing using 'long double'.", "We simulate magnetic fields and chaotic systems like the Lorenz model.", "Physics simulations using RK4 (Runge-Kutta 4th order) are the core of this workspace.", "Numerical precision is key: we use 80-bit or 128-bit floating point math."},
		},
		{
			Tag:       "deeplearning",
			Patterns:  []string{"deep learning", "dnn", "neural network", "goroutine parallel", "backpropagation", "layers", "weights", "bias", "training", "epochs"},
			Responses: []string{"I use Goroutines to parallelize neuron calculations for maximum speed.", "Our DNNs are implemented from scratch in Go and C without external libraries.", "We focus on high-precision numerical implementations of deep learning models.", "This model uses a 3-hidden-layer architecture for advanced feature extraction."},
		},
		{
			Tag:       "finance",
			Patterns:  []string{"finance", "bep", "break even point", "profitability", "sales", "safety margin", "contribution margin"},
			Responses: []string{"I can calculate the Break-Even Point (BEP) for business analysis.", "We model sales, fixed costs, and variable costs using DNN-based predictions.", "The safety margin and contribution margin ratio are key metrics we track."},
		},
		{
			Tag:       "math",
			Patterns:  []string{"math", "precision", "long double", "floating point", "accuracy", "high precision"},
			Responses: []string{"We use 'long double' in C for maximum accuracy in simulations.", "High-precision math prevents rounding errors in long-running chaotic models.", "The Biot-Savart calculations here are performed with 80-bit precision on win32."},
		},
		{
			Tag:       "project",
			Patterns:  []string{"what is this project", "workspace info", "repository info", "owner", "author", "creator", "shibata"},
			Responses: []string{"This is the High-Precision Scientific Computing & Deep Learning Workspace by Atsushi Shibata.", "It combines C, Go, and Python for advanced simulations and AI research.", "The repository is a collection of scratch-built tools for high-end computation."},
		},
		{
			Tag:       "testing",
			Patterns:  []string{"test", "npm test", "api test", "server test", "verify"},
			Responses: []string{"You can run 'npm test' to verify the Node.js server and simulation endpoints.", "Our test suite ensures all 50+ simulations are responding correctly.", "The batch-run endpoint is recently fixed and fully functional."},
		},
		{
			Tag:       "help",
			Patterns:  []string{"help", "what can you do", "who are you", "manual", "commands", "options", "usage"},
			Responses: []string{"I am a maximum-depth DNN chatbot built from scratch in Go.", "Ask me about science, math, finance, deep learning, or the project status.", "I use three hidden layers and parallel processing for intent classification."},
		},
	}

	// Prepare Vocabulary and Tags
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

	// Initialize and Train DNN (Now with 3 hidden layers)
	fmt.Println("--- Ultimate Conversational AI Training (Deep DNN + Goroutines) ---")
	fmt.Printf("Input Size: %d, H1: %d, H2: %d, H3: %d, Output: %d\n", len(allWords), HiddenSize1, HiddenSize2, HiddenSize3, len(intents))
	dnn := NewDNN(len(allWords), HiddenSize1, HiddenSize2, HiddenSize3, len(intents))

	startTime := time.Now()
	for epoch := 1; epoch <= Epochs; epoch++ {
		for _, data := range training {
			dnn.Train(data.Input, data.Target)
		}
		if epoch%1000 == 0 {
			fmt.Printf("Epoch %d/%d completed...\n", epoch, Epochs)
		}
	}
	fmt.Printf("Training completed in %v\n", time.Since(startTime))

	// Interactive Loop
	fmt.Println("\n--- Ultimate Chatbot Ready! (Type 'quit' to exit) ---")
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}
		userInput := scanner.Text()
		if strings.ToLower(userInput) == "quit" {
			break
		}

		tokens := tokenize(userInput)
		inputVec := bagOfWords(tokens, allWords)
		prediction := dnn.Forward(inputVec)

		// Find intent with highest probability
		maxIdx := 0
		maxVal := -1.0
		for i, val := range prediction {
			if val > maxVal {
				maxVal = val
				maxIdx = i
			}
		}

		if maxVal > 0.4 { // Slightly lower threshold for deeper network complexity
			intent := intents[maxIdx]
			response := intent.Responses[rand.Intn(len(intent.Responses))]
			fmt.Printf("AI: %s (Intent: %s, Confidence: %.2f%%)\n", response, intent.Tag, maxVal*100)
		} else {
			fmt.Println("AI: I'm not sure I understand what you mean. (Low Confidence)")
		}
	}
}
