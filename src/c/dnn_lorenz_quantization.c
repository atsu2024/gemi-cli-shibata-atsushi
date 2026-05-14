#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/**
 * dnn_lorenz_quantization.c
 * 
 * Deep Learning (DNN) version of the Lorenz Quantum Comparison Program.
 * Uses long double precision for high-fidelity scientific computation.
 * 
 * This program:
 * 1. Generates data from Classical and Semi-classical (Quantum-corrected) Lorenz systems.
 * 2. Trains a Deep Neural Network to learn the mapping from classical to quantum states.
 * 3. Compares the DNN prediction with the actual semi-classical values.
 */

/* Lorenz Parameters from "数値比較プログラム 量子化.txt" */
#define SIGMA 10.0L
#define RHO   5000.0L
#define BETA  (8.0L/3.0L)
#define HBAR  0.05L

/* DNN Configuration */
#define INPUT_DIM 3
#define HIDDEN_DIM 16
#define OUTPUT_DIM 3
#define LEARNING_RATE 0.001L
#define EPOCHS 20000
#define TRAIN_SAMPLES 1000

typedef struct {
    int num_layers;
    int *layer_sizes;
    long double **nodes;
    long double ***weights;
    long double **biases;
    long double **deltas;
} DNN;

/* Activation: Tanh for hidden layers, Linear for output (regression) */
long double activation_hidden(long double x) {
    return tanhl(x);
}

long double activation_derivative_hidden(long double x) {
    long double t = tanhl(x);
    return 1.0L - t * t;
}

/* Lorenz Physics Models */
void lorenz_classical(long double x, long double y, long double z, long double *dx, long double *dy, long double *dz) {
    *dx = SIGMA * (y - x);
    *dy = x * (RHO - z) - y;
    *dz = x * y - BETA * z;
}

void lorenz_semiclassical(long double x, long double y, long double z, long double *dx, long double *dy, long double *dz) {
    long double qc = (HBAR * HBAR) / 12.0L;
    *dx = SIGMA * (y - x);
    *dy = x * (RHO - z) - y - qc;
    *dz = x * y - BETA * z;
}

void rk4_step(void (*f)(long double,long double,long double,long double*,long double*,long double*), 
              long double *x, long double *y, long double *z, long double dt) {
    long double k1x,k1y,k1z, k2x,k2y,k2z, k3x,k3y,k3z, k4x,k4y,k4z;
    f(*x, *y, *z, &k1x, &k1y, &k1z);
    f(*x+0.5L*dt*k1x, *y+0.5L*dt*k1y, *z+0.5L*dt*k1z, &k2x, &k2y, &k2z);
    f(*x+0.5L*dt*k2x, *y+0.5L*dt*k2y, *z+0.5L*dt*k2z, &k3x, &k3y, &k3z);
    f(*x+dt*k3x, *y+dt*k3y, *z+dt*k3z, &k4x, &k4y, &k4z);
    *x += (dt/6.0L)*(k1x + 2*k2x + 2*k3x + k4x);
    *y += (dt/6.0L)*(k1y + 2*k2y + 2*k3y + k4y);
    *z += (dt/6.0L)*(k1z + 2*k2z + 2*k3z + k4z);
}

/* DNN Functions */
DNN* create_dnn(int num_layers, int *layer_sizes) {
    DNN *dnn = (DNN*)malloc(sizeof(DNN));
    dnn->num_layers = num_layers;
    dnn->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    memcpy(dnn->layer_sizes, layer_sizes, num_layers * sizeof(int));

    dnn->nodes = (long double**)malloc(num_layers * sizeof(long double*));
    dnn->biases = (long double**)malloc(num_layers * sizeof(long double*));
    dnn->deltas = (long double**)malloc(num_layers * sizeof(long double*));
    dnn->weights = (long double***)malloc((num_layers - 1) * sizeof(long double**));

    for (int i = 0; i < num_layers; i++) {
        dnn->nodes[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
        dnn->deltas[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
        if (i > 0) {
            dnn->biases[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
            dnn->weights[i-1] = (long double**)malloc(layer_sizes[i-1] * sizeof(long double*));
            long double limit = sqrtl(6.0L / (long double)(layer_sizes[i-1] + layer_sizes[i]));
            for (int j = 0; j < layer_sizes[i-1]; j++) {
                dnn->weights[i-1][j] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
                for (int k = 0; k < layer_sizes[i]; k++) {
                    dnn->weights[i-1][j][k] = ((long double)rand() / RAND_MAX) * 2.0L * limit - limit;
                }
            }
        }
    }
    return dnn;
}

void forward_prop(DNN *dnn, long double *inputs) {
    for (int i = 0; i < dnn->layer_sizes[0]; i++) dnn->nodes[0][i] = inputs[i];
    for (int i = 1; i < dnn->num_layers; i++) {
        for (int j = 0; j < dnn->layer_sizes[i]; j++) {
            long double sum = dnn->biases[i][j];
            for (int k = 0; k < dnn->layer_sizes[i-1]; k++) {
                sum += dnn->nodes[i-1][k] * dnn->weights[i-1][k][j];
            }
            if (i == dnn->num_layers - 1) dnn->nodes[i][j] = sum; // Linear output
            else dnn->nodes[i][j] = activation_hidden(sum);
        }
    }
}

void back_prop(DNN *dnn, long double *targets, long double lr) {
    int last = dnn->num_layers - 1;
    // Output layer error
    for (int i = 0; i < dnn->layer_sizes[last]; i++) {
        dnn->deltas[last][i] = targets[i] - dnn->nodes[last][i];
    }
    // Hidden layer errors
    for (int i = last - 1; i > 0; i--) {
        for (int j = 0; j < dnn->layer_sizes[i]; j++) {
            long double error = 0.0L;
            for (int k = 0; k < dnn->layer_sizes[i+1]; k++) {
                error += dnn->deltas[i+1][k] * dnn->weights[i][j][k];
            }
            dnn->deltas[i][j] = error * activation_derivative_hidden(dnn->nodes[i][j]);
        }
    }
    // Update weights and biases
    for (int i = 1; i < dnn->num_layers; i++) {
        for (int j = 0; j < dnn->layer_sizes[i]; j++) {
            dnn->biases[i][j] += lr * dnn->deltas[i][j];
            for (int k = 0; k < dnn->layer_sizes[i-1]; k++) {
                dnn->weights[i-1][k][j] += lr * dnn->deltas[i][j] * dnn->nodes[i-1][k];
            }
        }
    }
}

int main() {
    srand((unsigned)time(NULL));
    printf("--- Lorenz Quantum-Correction DNN (Long Double Precision) ---\n\n");

    /* 1. Data Generation */
    printf("Generating %d trajectory samples...\n", TRAIN_SAMPLES);
    long double xc = 1.0L, yc = 1.0L, zc = 1.0L;
    long double xq = 1.0L, yq = 1.0L, zq = 1.0L;
    long double dt = 0.001L;

    long double **train_inputs = (long double**)malloc(TRAIN_SAMPLES * sizeof(long double*));
    long double **train_targets = (long double**)malloc(TRAIN_SAMPLES * sizeof(long double*));

    for (int i = 0; i < TRAIN_SAMPLES; i++) {
        train_inputs[i] = (long double*)malloc(3 * sizeof(long double));
        train_targets[i] = (long double*)malloc(3 * sizeof(long double));
        
        // Input: Classical state
        train_inputs[i][0] = xc;
        train_inputs[i][1] = yc;
        train_inputs[i][2] = zc;
        
        // Target: Semi-classical state (mapping to learn)
        train_targets[i][0] = xq;
        train_targets[i][1] = yq;
        train_targets[i][2] = zq;

        rk4_step(lorenz_classical, &xc, &yc, &zc, dt);
        rk4_step(lorenz_semiclassical, &xq, &yq, &zq, dt);
    }

    /* 2. Setup DNN */
    int layers[] = {INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM};
    DNN *dnn = create_dnn(4, layers);

    /* 3. Training Loop */
    printf("Training DNN for %d epochs...\n", EPOCHS);
    for (int e = 0; e <= EPOCHS; e++) {
        long double total_mse = 0;
        for (int i = 0; i < TRAIN_SAMPLES; i++) {
            forward_prop(dnn, train_inputs[i]);
            back_prop(dnn, train_targets[i], LEARNING_RATE);
            
            for (int j = 0; j < OUTPUT_DIM; j++) {
                long double diff = train_targets[i][j] - dnn->nodes[dnn->num_layers-1][j];
                total_mse += diff * diff;
            }
        }
        if (e % 2000 == 0) {
            printf("Epoch %5d | MSE: %.20Lf\n", e, total_mse / (TRAIN_SAMPLES * OUTPUT_DIM));
        }
    }

    /* 4. Final Comparison & Verification */
    printf("\n--- Validation (Actual vs Predicted Quantum-Corrected State) ---\n");
    printf("%-10s | %-20s | %-20s | %-20s\n", "Dim", "Classical", "Semi-Classical", "DNN Predicted");
    printf("-----------|----------------------|----------------------|----------------------\n");
    
    // Pick a sample from the end of training
    int test_idx = TRAIN_SAMPLES - 1;
    forward_prop(dnn, train_inputs[test_idx]);
    
    char *dims[] = {"X", "Y", "Z"};
    for (int j = 0; j < 3; j++) {
        printf("%-10s | %20.10Lf | %20.10Lf | %20.10Lf\n", 
               dims[j], train_inputs[test_idx][j], train_targets[test_idx][j], dnn->nodes[dnn->num_layers-1][j]);
    }

    long double final_err = sqrtl(
        powl(train_targets[test_idx][0] - dnn->nodes[dnn->num_layers-1][0], 2.0L) +
        powl(train_targets[test_idx][1] - dnn->nodes[dnn->num_layers-1][1], 2.0L) +
        powl(train_targets[test_idx][2] - dnn->nodes[dnn->num_layers-1][2], 2.0L)
    );
    printf("\nFinal Euclidean Error in Prediction: %.20Lf\n", final_err);

    return 0;
}
