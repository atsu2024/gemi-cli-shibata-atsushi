package main

import (
	"fmt"
	"os"
	"os/exec"
	"sync"
)

func runSimulation(target string, wg *sync.WaitGroup) {
	defer wg.Done()

	fmt.Printf("[Go] Starting simulation for target: %s\n", target)
	cmd := exec.Command("./break_even_pid.exe", target)
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("[Go] Error running simulation for target %s: %v\n", target, err)
		return
	}
	fmt.Printf("[Go] Result for target %s:\n%s\n", target, string(output))
}

func main() {
	// 1. Build the C program
	fmt.Println("[Go] Building C program...")
	buildCmd := exec.Command("gcc", "-o", "break_even_pid.exe", "break_even_pid.c")
	buildOutput, err := buildCmd.CombinedOutput()
	if err != nil {
		fmt.Printf("[Go] Build failed: %v\n%s\n", err, string(buildOutput))
		os.Exit(1)
	}
	fmt.Println("[Go] Build successful.")

	// 2. Run simulations in parallel using goroutines
	targets := []string{"100.0", "500.0", "1000.0"}
	var wg sync.WaitGroup

	for _, t := range targets {
		wg.Add(1)
		go runSimulation(t, &wg)
	}

	wg.Wait()
	fmt.Println("[Go] All simulations completed.")
}
