#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/**
 * dl_long_double.c
 * 
 * A generalized Deep Learning (MLP) implementation using long double precision.
 * Features:
 * - Dynamic layer configuration
 * - CSV data loading
 * - High-precision backpropagation
 */

typedef struct {
    int num_layers;
    int *layer_sizes;
    long double **nodes;
    long double **biases;
    long double ***weights;
    long double **deltas;
} NeuralNetwork;

// Activation function: Sigmoid
long double sigmoid_ld(long double x) {
    return 1.0L / (1.0L + expl(-x));
}

// Derivative of Sigmoid
long double sigmoid_derivative_ld(long double x) {
    return x * (1.0L - x);
}

// Create and initialize the network
NeuralNetwork* create_network(int num_layers, int *layer_sizes) {
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    memcpy(nn->layer_sizes, layer_sizes, num_layers * sizeof(int));

    nn->nodes = (long double**)malloc(num_layers * sizeof(long double*));
    nn->biases = (long double**)malloc(num_layers * sizeof(long double*));
    nn->deltas = (long double**)malloc(num_layers * sizeof(long double*));
    nn->weights = (long double***)malloc((num_layers - 1) * sizeof(long double**));

    for (int i = 0; i < num_layers; i++) {
        nn->nodes[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
        nn->deltas[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
        if (i > 0) {
            nn->biases[i] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
            nn->weights[i-1] = (long double**)malloc(layer_sizes[i-1] * sizeof(long double*));
            for (int j = 0; j < layer_sizes[i-1]; j++) {
                nn->weights[i-1][j] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
                for (int k = 0; k < layer_sizes[i]; k++) {
                    // Xavier/Glorot initialization
                    long double limit = sqrtl(6.0L / (layer_sizes[i-1] + layer_sizes[i]));
                    nn->weights[i-1][j][k] = ((long double)rand() / RAND_MAX) * 2.0L * limit - limit;
                }
            }
            for (int j = 0; j < layer_sizes[i]; j++) {
                nn->biases[i][j] = 0.0L;
            }
        }
    }
    return nn;
}

// Forward propagation
void forward_prop(NeuralNetwork *nn, long double *inputs) {
    for (int i = 0; i < nn->layer_sizes[0]; i++) {
        nn->nodes[0][i] = inputs[i];
    }
    for (int i = 1; i < nn->num_layers; i++) {
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            long double activation = nn->biases[i][j];
            for (int k = 0; k < nn->layer_sizes[i-1]; k++) {
                activation += nn->nodes[i-1][k] * nn->weights[i-1][k][j];
            }
            nn->nodes[i][j] = sigmoid_ld(activation);
        }
    }
}

// Backward propagation and weight update
void back_prop(NeuralNetwork *nn, long double *targets, long double lr) {
    int last = nn->num_layers - 1;
    
    // Output layer error
    for (int i = 0; i < nn->layer_sizes[last]; i++) {
        long double error = targets[i] - nn->nodes[last][i];
        nn->deltas[last][i] = error * sigmoid_derivative_ld(nn->nodes[last][i]);
    }

    // Hidden layers error
    for (int i = last - 1; i > 0; i--) {
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            long double error = 0.0L;
            for (int k = 0; k < nn->layer_sizes[i+1]; k++) {
                error += nn->deltas[i+1][k] * nn->weights[i][j][k];
            }
            nn->deltas[i][j] = error * sigmoid_derivative_ld(nn->nodes[i][j]);
        }
    }

    // Update weights and biases
    for (int i = 1; i < nn->num_layers; i++) {
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            nn->biases[i][j] += lr * nn->deltas[i][j];
            for (int k = 0; k < nn->layer_sizes[i-1]; k++) {
                nn->weights[i-1][k][j] += lr * nn->deltas[i][j] * nn->nodes[i-1][k];
            }
        }
    }
}

// Load CSV data
int load_csv(const char *filename, long double ***inputs, long double ***targets, int in_dim, int tar_dim) {
    FILE *file = fopen(filename, "r");
    if (!file) return -1;

    char line[4096];
    int count = 0;
    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) > 5) count++; // Simple check for non-empty line
    }
    rewind(file);

    *inputs = (long double**)malloc(count * sizeof(long double*));
    *targets = (long double**)malloc(count * sizeof(long double*));

    for (int i = 0; i < count; i++) {
        (*inputs)[i] = (long double*)malloc(in_dim * sizeof(long double));
        (*targets)[i] = (long double*)malloc(tar_dim * sizeof(long double));
        if (!fgets(line, sizeof(line), file)) break;

        char *token = strtok(line, ",");
        for (int j = 0; j < in_dim && token; j++) {
            (*inputs)[i][j] = strtold(token, NULL);
            token = strtok(NULL, ",");
        }
        for (int j = 0; j < tar_dim && token; j++) {
            (*targets)[i][j] = strtold(token, NULL);
            token = strtok(NULL, ",");
        }
    }
    fclose(file);
    return count;
}

int main() {
    srand(time(NULL));

    // Configuration
    const char *csv_file = "xor_data.csv";
    int input_dim = 2;
    int target_dim = 1;
    int layers[] = {input_dim, 8, 8, target_dim}; // Deep Network
    int num_layers = sizeof(layers) / sizeof(layers[0]);

    long double **train_in, **train_tar;
    int num_samples = load_csv(csv_file, &train_in, &train_tar, input_dim, target_dim);

    if (num_samples <= 0) {
        printf("Error: Data file '%s' not found or empty.\n", csv_file);
        return 1;
    }

    NeuralNetwork *nn = create_network(num_layers, layers);
    long double learning_rate = 0.5L;
    int epochs = 100000;

    printf("=== Deep Learning (long double) ===\n");
    printf("Samples: %d, Epochs: %d, LR: %.2Lf\n", num_samples, epochs, learning_rate);

    for (int e = 0; e <= epochs; e++) {
        for (int i = 0; i < num_samples; i++) {
            forward_prop(nn, train_in[i]);
            back_prop(nn, train_tar[i], learning_rate);
        }
        if (e % 10000 == 0) {
            long double total_err = 0;
            for (int i = 0; i < num_samples; i++) {
                forward_prop(nn, train_in[i]);
                total_err += fabsl(train_tar[i][0] - nn->nodes[num_layers - 1][0]);
            }
            printf("Epoch %6d | Avg Error: %.15Lf\n", e, total_err / num_samples);
        }
    }

    printf("\n--- Final Predictions ---\n");
    for (int i = 0; i < num_samples; i++) {
        forward_prop(nn, train_in[i]);
        printf("In: (%.0Lf, %.0Lf) -> Target: %.0Lf | Predicted: %.15Lf\n", 
               train_in[i][0], train_in[i][1], train_tar[i][0], nn->nodes[num_layers-1][0]);
    }

    return 0;
}
