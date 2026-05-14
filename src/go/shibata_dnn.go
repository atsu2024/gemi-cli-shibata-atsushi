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
	"sync"
	"time"
)

// DNN Parameters
const (
	InputSize  = 1024 // 32x32
	HiddenSize = 128
	OutputSize = 1024
	LearningRate = 0.05
	Epochs       = 200
)

// Activation Functions
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Autoencoder structure
type Autoencoder struct {
	w1 [][]float64
	b1 []float64
	h  []float64
	w2 [][]float64
	b2 []float64
	o  []float64
}

func NewAutoencoder() *Autoencoder {
	rand.Seed(time.Now().UnixNano())
	ae := &Autoencoder{
		w1: make([][]float64, InputSize),
		b1: make([]float64, HiddenSize),
		h:  make([]float64, HiddenSize),
		w2: make([][]float64, HiddenSize),
		b2: make([]float64, OutputSize),
		o:  make([]float64, OutputSize),
	}

	for i := 0; i < InputSize; i++ {
		ae.w1[i] = make([]float64, HiddenSize)
		for j := 0; j < HiddenSize; j++ {
			ae.w1[i][j] = (rand.Float64()*2 - 1) * math.Sqrt(1.0/InputSize)
		}
	}

	for i := 0; i < HiddenSize; i++ {
		ae.w2[i] = make([]float64, OutputSize)
		for j := 0; j < OutputSize; j++ {
			ae.w2[i][j] = (rand.Float64()*2 - 1) * math.Sqrt(1.0/HiddenSize)
		}
	}
	return ae
}

func (ae *Autoencoder) Forward(input []float64) {
	// Input to Hidden
	var wg sync.WaitGroup
	wg.Add(HiddenSize)
	for j := 0; j < HiddenSize; j++ {
		go func(idx int) {
			defer wg.Done()
			sum := ae.b1[idx]
			for i := 0; i < InputSize; i++ {
				sum += input[i] * ae.w1[i][idx]
			}
			ae.h[idx] = sigmoid(sum)
		}(j)
	}
	wg.Wait()

	// Hidden to Output
	wg.Add(OutputSize)
	for j := 0; j < OutputSize; j++ {
		go func(idx int) {
			defer wg.Done()
			sum := ae.b2[idx]
			for i := 0; i < HiddenSize; i++ {
				sum += ae.h[i] * ae.w2[i][idx]
			}
			ae.o[idx] = sigmoid(sum)
		}(j)
	}
	wg.Wait()
}

func (ae *Autoencoder) Train(input []float64) {
	ae.Forward(input)

	// Output errors
	outDeltas := make([]float64, OutputSize)
	for i := 0; i < OutputSize; i++ {
		outDeltas[i] = (input[i] - ae.o[i]) * sigmoidDerivative(ae.o[i])
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
				err += outDeltas[j] * ae.w2[idx][j]
			}
			hDeltas[idx] = err * sigmoidDerivative(ae.h[idx])
		}(i)
	}
	wg.Wait()

	// Update weights
	wg.Add(HiddenSize)
	for i := 0; i < HiddenSize; i++ {
		go func(idx int) {
			defer wg.Done()
			for j := 0; j < OutputSize; j++ {
				ae.w2[idx][j] += LearningRate * outDeltas[j] * ae.h[idx]
			}
		}(i)
	}
	for j := 0; j < OutputSize; j++ {
		ae.b2[j] += LearningRate * outDeltas[j]
	}
	wg.Wait()

	wg.Add(InputSize)
	for i := 0; i < InputSize; i++ {
		go func(idx int) {
			defer wg.Done()
			for j := 0; j < HiddenSize; j++ {
				ae.w1[idx][j] += LearningRate * hDeltas[j] * input[idx]
			}
		}(i)
	}
	for j := 0; j < HiddenSize; j++ {
		ae.b1[j] += LearningRate * hDeltas[j]
	}
	wg.Wait()
}

func main() {
	path := `C:\Users\S111478\OneDrive - 三機工業株式会社\ドキュメント\shibata_atsushi01.png`
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Could not open image: %v. Trying fallback...\n", err)
		path = `C:\Users\S111478\OneDrive - 三機工業株式会社\ドキュメント\shibata_atsushi.jpeg`
		file, err = os.Open(path)
		if err != nil {
			fmt.Printf("Error opening fallback image: %v\n", err)
			return
		}
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		fmt.Printf("Error decoding image: %v\n", err)
		return
	}

	fmt.Printf("Processing image: %s\n", path)
	
	// Pre-process: grayscale and resize to 32x32
	inputData := make([]float64, InputSize)
	bounds := img.Bounds()
	w, h := bounds.Max.X, bounds.Max.Y
	
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			// Simple sampling
			imgX := x * w / 32
			imgY := y * h / 32
			r, g, b, _ := img.At(imgX, imgY).RGBA()
			gray := (0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 65535.0
			inputData[y*32+x] = gray
		}
	}

	ae := NewAutoencoder()
	fmt.Println("Starting DNN (Autoencoder) training with Goroutines...")
	start := time.Now()
	for epoch := 1; epoch <= Epochs; epoch++ {
		ae.Train(inputData)
		if epoch%20 == 0 || epoch == 1 {
			mse := 0.0
			for i := 0; i < InputSize; i++ {
				mse += math.Pow(inputData[i]-ae.o[i], 2)
			}
			fmt.Printf("Epoch %d/%d, MSE: %.6f\n", epoch, Epochs, mse/InputSize)
		}
	}
	fmt.Printf("Training completed in %v\n", time.Since(start))

	// Generate output image
	outImg := image.NewGray(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			val := uint8(ae.o[y*32+x] * 255)
			outImg.SetGray(x, y, color.Gray{Y: val})
		}
	}

	outPath := "shibata_dnn_output.png"
	outFile, err := os.Create(outPath)
	if err != nil {
		fmt.Printf("Error creating output file: %v\n", err)
		return
	}
	defer outFile.Close()
	png.Encode(outFile, outImg)

	fmt.Printf("DNN processed image saved to %s\n", outPath)
}
