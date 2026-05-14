# Project Overview: High-Precision Scientific Computing & Deep Learning

This workspace is a specialized environment for high-precision numerical simulations and deep learning experiments. It combines a Node.js-based Express server with a large collection of C-language programs focused on physics simulations and neural network implementations from scratch.

## Key Components

### 1. Scientific Simulations (C)
A series of C programs that utilize `long double` for maximum numerical precision.
- **Physics Models:**
  - `biot_savart_precision.c`: High-precision magnetic field calculations using the Biot-Savart law.
  - `lorenz_dnn_final.c`: Semi-classical Lorenz system simulations using RK4 (Runge-Kutta 4th order) and Deep Neural Networks.
- **Mathematical Experiments:**
  - `cell_system.c`, `emoney_jpy_precision.c`: Various precision-focused modeling scripts.

### 2. Deep Learning from Scratch (C & Go)
Implementations of neural networks without external ML libraries (like TensorFlow or PyTorch), focusing on foundational understanding and precision.
- **C Models:** DNN (Deep Neural Networks), MLP (Multi-Layer Perceptrons).
- **Go Models:** Parallel implementations using Goroutines for increased performance.
- **Core Files:** `src/c/deep_mlp_long_double.c`, `src/go/deeplearning_dnn_goroutine.go`.
- **Binaries:** All compiled programs are located in the `bin/` directory.

### 3. Centralized Binaries
All compiled simulations and analysis tools are stored in the `bin/` directory for easy access.
- **C Binaries:** `bin/*.exe`
- **Go Binaries:** `bin/*_go.exe` and `bin/*_root.exe`

### 4. Web Service (Node.js)
A basic Express server intended for potential API integration or serving simulation results.
- **Entry Point:** `server.js`
- **Technologies:** Express, Axios, Dotenv.

## Building and Running

### Node.js Server
- **Start:** `npm start` (runs `node server.js`)
- **Port:** Defaults to `3000` (configurable via `.env`).

### C Programs
The C programs are standalone scripts. They can be compiled using a C compiler like `gcc`.
- **Compile Example:** `gcc -o biot_savart_precision biot_savart_precision.c -lm`
- **Note:** Many files require the math library (`-lm`) and some use MinGW-specific defines (e.g., `__USE_MINGW_ANSI_STDIO`).

## Development Conventions
- **Precision:** Use `long double` and corresponding math functions (`sqrtl`, `tanhl`, `powl`, `fabsl`) for all scientific calculations.
- **Neural Networks:** Prefer explicit memory management for weight matrices and node layers.
- **Data Storage:** Weights are typically serialized to `.bin` files for efficiency, while results are exported to `.csv` for analysis.
