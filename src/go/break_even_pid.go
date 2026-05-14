package main

import (
	"fmt"
	"math"
	"sync"
)

// PIDController represents a PID controller using float64 (high precision)
type PIDController struct {
	Kp, Ki, Kd float64
	Integral   float64
	PrevError  float64
}

func (pid *PIDController) Calculate(setpoint, pv, dt float64) float64 {
	error := setpoint - pv
	pid.Integral += error * dt
	derivative := (error - pid.PrevError) / dt
	output := pid.Kp*error + pid.Ki*pid.Integral + pid.Kd*derivative
	pid.PrevError = error
	return output
}

// System simulates the "Grand Strategy Map" logic: output = input * 10 + 1
func System(input float64) float64 {
	return input*10.0 + 1.0
}

// SimulationTask represents a single competitive bidding simulation
type SimulationTask struct {
	BidID  int
	Target float64 // Break-even point in 100m units
}

type Result struct {
	BidID  int
	Target float64
	Input  float64
	Output float64
}

var (
	results []Result
	mu      sync.Mutex
)

func runSimulation(wg *sync.WaitGroup, task SimulationTask) {
	defer wg.Done()

	pid := PIDController{Kp: 0.01, Ki: 0.005, Kd: 0.001}
	currentU := 0.0
	dt := 0.1
	maxSteps := 500 // Increased for higher precision

	for i := 0; i < maxSteps; i++ {
		y := System(currentU)
		error := task.Target - y

		if math.Abs(error) < 1e-6 {
			mu.Lock()
			results = append(results, Result{task.BidID, task.Target, currentU, y})
			mu.Unlock()
			return
		}

		deltaU := pid.Calculate(task.Target, y, dt)
		currentU += deltaU
	}

	mu.Lock()
	results = append(results, Result{task.BidID, task.Target, currentU, System(currentU)})
	mu.Unlock()
}

func main() {
	fmt.Println("EcoSphere - Atsushi Shibata: Break-even PID Calculation (100m units)")
	fmt.Println("System: Grand Strategy Map (Output = Input * 10 + 1)")
	fmt.Println("Running competitive bidding simulations using Goroutines...\n")

	targets := []float64{100.0, 150.0, 200.0, 250.0, 300.0}

	var wg sync.WaitGroup
	for i, target := range targets {
		wg.Add(1)
		task := SimulationTask{i + 1, target}
		go runSimulation(&wg, task)
	}

	wg.Wait()

	fmt.Println("---------------------------------------------------------------")
	fmt.Printf("%-10s | %-15s | %-15s | %-15s\n", "Bid ID", "Target (100m)", "Input (PID)", "Output (Actual)")
	fmt.Println("---------------------------------------------------------------")
	for _, res := range results {
		fmt.Printf("%-10d | %-15.2f | %-15.6f | %-15.6f\n", res.BidID, res.Target, res.Input, res.Output)
	}
	fmt.Println("---------------------------------------------------------------")
	fmt.Println("All simulations completed.")
}
