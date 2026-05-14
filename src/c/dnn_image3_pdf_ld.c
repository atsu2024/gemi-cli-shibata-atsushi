#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/*
 * dnn_image3_pdf_ld.c
 * Deep Neural Network (Autoencoder) for processing image data related to image-3.pdf.
 * Uses long double precision for high accuracy calculations.
 */

typedef struct {
    int num_layers;
    int *layer_sizes;
    long double **nodes;
    long double ***weights;
    long double **biases;
    long double **deltas;
} DeepMLP;

long double sigmoid_ld(long double x) {
    if (x > 100.0L) return 1.0L;
    if (x < -100.0L) return 0.0L;
    return 1.0L / (1.0L + expl(-x));
}

long double sigmoid_derivative_ld(long double x) {
    return x * (1.0L - x);
}

long double random_xavier(int fan_in, int fan_out) {
    long double range = sqrtl(6.0L / (long double)(fan_in + fan_out));
    return ((long double)rand() / RAND_MAX) * 2.0L * range - range;
}

DeepMLP* create_mlp(int num_layers, int *layer_sizes) {
    DeepMLP *mlp = (DeepMLP*)malloc(sizeof(DeepMLP));
    mlp->num_layers = num_layers;
    mlp->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; i++) mlp->layer_sizes[i] = layer_sizes[i];

    mlp->nodes = (long double**)malloc(num_layers * sizeof(long double*));
    mlp->biases = (long double**)malloc(num_layers * sizeof(long double*));
    mlp->deltas = (long double**)malloc(num_layers * sizeof(long double*));
    mlp->weights = (long double***)malloc((num_layers - 1) * sizeof(long double**));

    for (int i = 0; i < num_layers; i++) {
        mlp->nodes[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
        mlp->deltas[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
        if (i > 0) {
            mlp->biases[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
            mlp->weights[i-1] = (long double**)malloc(layer_sizes[i-1] * sizeof(long double*));
            for (int j = 0; j < layer_sizes[i-1]; j++) {
                mlp->weights[i-1][j] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
                for (int k = 0; k < layer_sizes[i]; k++) {
                    mlp->weights[i-1][j][k] = random_xavier(layer_sizes[i-1], layer_sizes[i]);
                }
            }
        }
    }
    return mlp;
}

void forward_prop(DeepMLP *mlp, long double *inputs) {
    for (int i = 0; i < mlp->layer_sizes[0]; i++) mlp->nodes[0][i] = inputs[i];
    for (int i = 1; i < mlp->num_layers; i++) {
        for (int j = 0; j < mlp->layer_sizes[i]; j++) {
            long double activation = mlp->biases[i][j];
            for (int k = 0; k < mlp->layer_sizes[i-1]; k++) {
                activation += mlp->nodes[i-1][k] * mlp->weights[i-1][k][j];
            }
            mlp->nodes[i][j] = sigmoid_ld(activation);
        }
    }
}

void back_prop(DeepMLP *mlp, long double *targets, long double lr) {
    int last = mlp->num_layers - 1;
    for (int i = 0; i < mlp->layer_sizes[last]; i++) {
        long double error = targets[i] - mlp->nodes[last][i];
        mlp->deltas[last][i] = error * sigmoid_derivative_ld(mlp->nodes[last][i]);
    }
    for (int i = last - 1; i > 0; i--) {
        for (int j = 0; j < mlp->layer_sizes[i]; j++) {
            long double error = 0.0L;
            for (int k = 0; k < mlp->layer_sizes[i+1]; k++) {
                error += mlp->deltas[i+1][k] * mlp->weights[i][j][k];
            }
            mlp->deltas[i][j] = error * sigmoid_derivative_ld(mlp->nodes[i][j]);
        }
    }
    for (int i = 1; i < mlp->num_layers; i++) {
        for (int j = 0; j < mlp->layer_sizes[i]; j++) {
            mlp->biases[i][j] += lr * mlp->deltas[i][j];
            for (int k = 0; k < mlp->layer_sizes[i-1]; k++) {
                mlp->weights[i-1][k][j] += lr * mlp->deltas[i][j] * mlp->nodes[i-1][k];
            }
        }
    }
}

int main() {
    srand((unsigned int)time(NULL));
    int layer_sizes[] = {1024, 256, 64, 256, 1024};
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    DeepMLP *mlp = create_mlp(num_layers, layer_sizes);

    printf("Deep Neural Network for image-3.pdf initialized.\n");
    printf("Input Layer: %d nodes\n", layer_sizes[0]);
    printf("Output Layer: %d nodes\n", layer_sizes[num_layers-1]);
    printf("Precision: long double\n\n");

    long double *input_data = (long double*)malloc(1024 * sizeof(long double));
    // Simulate reading data from image-3.pdf (32x32 QR code pattern)
    for (int i = 0; i < 1024; i++) {
        input_data[i] = (rand() % 2 == 0) ? 1.0L : 0.0L;
    }

    long double lr = 0.1L;
    int epochs = 1000;
    
    printf("Starting training (Autoencoder Mode)...\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        forward_prop(mlp, input_data);
        back_prop(mlp, input_data, lr);
        
        if (epoch % 100 == 0) {
            long double mse = 0.0L;
            for (int i = 0; i < 1024; i++) {
                long double diff = input_data[i] - mlp->nodes[num_layers-1][i];
                mse += diff * diff;
            }
            mse /= 1024.0L;
            printf("Epoch %d, MSE: %Le\n", epoch, mse);
        }
    }

    printf("\nTraining complete.\n");
    printf("Reconstruction Verification (First 10 nodes):\n");
    forward_prop(mlp, input_data);
    for (int i = 0; i < 10; i++) {
        printf("Node %d: Target %.1Lf, Output %Le\n", i, input_data[i], mlp->nodes[num_layers-1][i]);
    }

    return 0;
}
