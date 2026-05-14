package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// DNN Parameters
const (
	InputSize    = 3   // Month, Day, DayOfWeek
	HiddenSize   = 16
	OutputSize   = 1   // IsHoliday (0 to 1)
	LearningRate = 0.1
	Epochs       = 1000
)

// Activation Functions
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// DNN structure
type DNN struct {
	w1 [][]float64
	b1 []float64
	h  []float64
	w2 [][]float64
	b2 []float64
	o  []float64
}

func NewDNN() *DNN {
	rand.Seed(time.Now().UnixNano())
	dnn := &DNN{
		w1: make([][]float64, InputSize),
		b1: make([]float64, HiddenSize),
		h:  make([]float64, HiddenSize),
		w2: make([][]float64, HiddenSize),
		b2: make([]float64, OutputSize),
		o:  make([]float64, OutputSize),
	}

	for i := 0; i < InputSize; i++ {
		dnn.w1[i] = make([]float64, HiddenSize)
		for j := 0; j < HiddenSize; j++ {
			dnn.w1[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(InputSize))
		}
	}

	for i := 0; i < HiddenSize; i++ {
		dnn.w2[i] = make([]float64, OutputSize)
		for j := 0; j < OutputSize; j++ {
			dnn.w2[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(HiddenSize))
		}
	}
	return dnn
}

func (dnn *DNN) Forward(input []float64) {
	// Input to Hidden
	var wg sync.WaitGroup
	wg.Add(HiddenSize)
	for j := 0; j < HiddenSize; j++ {
		go func(idx int) {
			defer wg.Done()
			sum := dnn.b1[idx]
			for i := 0; i < InputSize; i++ {
				sum += input[i] * dnn.w1[i][idx]
			}
			dnn.h[idx] = sigmoid(sum)
		}(j)
	}
	wg.Wait()

	// Hidden to Output
	wg.Add(OutputSize)
	for j := 0; j < OutputSize; j++ {
		go func(idx int) {
			defer wg.Done()
			sum := dnn.b2[idx]
			for i := 0; i < HiddenSize; i++ {
				sum += dnn.h[i] * dnn.w2[i][idx]
			}
			dnn.o[idx] = sigmoid(sum)
		}(j)
	}
	wg.Wait()
}

func (dnn *DNN) Train(input []float64, target []float64) {
	dnn.Forward(input)

	// Output errors
	outDeltas := make([]float64, OutputSize)
	for i := 0; i < OutputSize; i++ {
		outDeltas[i] = (target[i] - dnn.o[i]) * sigmoidDerivative(dnn.o[i])
	}

	// Hidden errors
	hDeltas := make([]float64, HiddenSize)
	var wg sync.WaitGroup
	wg.Add(HiddenSize)
	for i := 0; i < HiddenSize; i++ {
		go func(idx int) {
			defer wg.Done()
			var err float64
			for j := 0; j < OutputSize; j++ {
				err += outDeltas[j] * dnn.w2[idx][j]
			}
			hDeltas[idx] = err * sigmoidDerivative(dnn.h[idx])
		}(i)
	}
	wg.Wait()

	// Update weights
	wg.Add(HiddenSize)
	for i := 0; i < HiddenSize; i++ {
		go func(idx int) {
			defer wg.Done()
			for j := 0; j < OutputSize; j++ {
				dnn.w2[idx][j] += LearningRate * outDeltas[j] * dnn.h[idx]
			}
		}(i)
	}
	for j := 0; j < OutputSize; j++ {
		dnn.b2[j] += LearningRate * outDeltas[j]
	}
	wg.Wait()

	wg.Add(InputSize)
	for i := 0; i < InputSize; i++ {
		go func(idx int) {
			defer wg.Done()
			for j := 0; j < HiddenSize; j++ {
				dnn.w1[idx][j] += LearningRate * hDeltas[j] * input[idx]
			}
		}(i)
	}
	for j := 0; j < HiddenSize; j++ {
		dnn.b1[j] += LearningRate * hDeltas[j]
	}
	wg.Wait()
}

// Holiday logic simulation (based on cal.py)
func isHoliday(year, month, day int) bool {
	// Fixed holidays
	fixedHolidays := map[[2]int]bool{
		{1, 1}:   true, {2, 11}:  true, {2, 23}:  true, {3, 20}:  true,
		{4, 29}:  true, {5, 3}:   true, {5, 4}:   true, {5, 5}:   true,
		{8, 11}:  true, {9, 22}:  true, {11, 3}:  true, {11, 23}: true,
	}
	if fixedHolidays[[2]int{month, day}] {
		return true
	}
	
	// Simplify for example: Check if Sunday
	t := time.Date(year, time.Month(month), day, 0, 0, 0, 0, time.Local)
	if t.Weekday() == time.Sunday {
		return true
	}
	return false
}

func main() {
	fmt.Println("Deep Learning Calendar (Holiday Prediction) in Go")
	fmt.Println("Initializing DNN with Goroutines...")

	dnn := NewDNN()

	// Generate Training Data (Year 2024)
	type DataPoint struct {
		input  []float64
		target []float64
	}
	var trainingData []DataPoint
	year := 2024
	for m := 1; m <= 12; m++ {
		for d := 1; d <= 28; d++ { // Use first 28 days for simplicity
			t := time.Date(year, time.Month(m), d, 0, 0, 0, 0, time.Local)
			target := 0.0
			if isHoliday(year, m, d) {
				target = 1.0
			}
			trainingData = append(trainingData, DataPoint{
				input:  []float64{float64(m) / 12.0, float64(d) / 31.0, float64(t.Weekday()) / 6.0},
				target: []float64{target},
			})
		}
	}

	fmt.Printf("Training on %d data points...\n", len(trainingData))
	start := time.Now()
	for epoch := 1; epoch <= Epochs; epoch++ {
		totalError := 0.0
		for _, dp := range trainingData {
			dnn.Train(dp.input, dp.target)
			totalError += math.Abs(dp.target[0] - dnn.o[0])
		}
		if epoch%100 == 0 {
			fmt.Printf("Epoch %d/%d, Avg Error: %.4f\n", epoch, Epochs, totalError/float64(len(trainingData)))
		}
	}
	fmt.Printf("Training completed in %v\n", time.Since(start))

	// Test prediction
	fmt.Println("\nPredictions for some dates in 2025:")
	testDates := []struct{ y, m, d int }{
		{2025, 1, 1},  // New Year (Holiday)
		{2025, 1, 5},  // Sunday (Holiday in our simplified logic)
		{2025, 1, 6},  // Monday (Not Holiday)
		{2025, 5, 5},  // Children's Day (Holiday)
	}

	for _, td := range testDates {
		t := time.Date(td.y, time.Month(td.m), td.d, 0, 0, 0, 0, time.Local)
		input := []float64{float64(td.m) / 12.0, float64(td.d) / 31.0, float64(t.Weekday()) / 6.0}
		dnn.Forward(input)
		prediction := dnn.o[0]
		status := "Workday"
		if prediction > 0.5 {
			status = "Holiday"
		}
		actual := "Workday"
		if isHoliday(td.y, td.m, td.d) {
			actual = "Holiday"
		}
		fmt.Printf("%d/%02d/%02d (WD:%d) -> Predicted: %.4f (%s), Actual: %s\n", 
			td.y, td.m, td.d, t.Weekday(), prediction, status, actual)
	}
}
