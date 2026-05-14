#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/**
 * deep_mlp_long_double.c
 * 
 * A generalized Multi-Layer Perceptron using long double for high-precision scientific deep learning.
 * Now with CSV loading capabilities.
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
    return 1.0L / (1.0L + expl(-x));
}

long double sigmoid_derivative_ld(long double x) {
    return x * (1.0L - x);
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
            mlp->biases[i] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
            mlp->weights[i-1] = (long double**)malloc(layer_sizes[i-1] * sizeof(long double*));
            for (int j = 0; j < layer_sizes[i-1]; j++) {
                mlp->weights[i-1][j] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
                for (int k = 0; k < layer_sizes[i]; k++) {
                    mlp->weights[i-1][j][k] = ((long double)rand() / RAND_MAX) * 2.0L - 1.0L;
                }
            }
            for (int j = 0; j < layer_sizes[i]; j++) {
                mlp->biases[i][j] = ((long double)rand() / RAND_MAX) * 2.0L - 1.0L;
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

// Load training data from CSV
int load_csv(const char *filename, long double ***inputs, long double ***targets, int input_dim, int target_dim) {
    FILE *file = fopen(filename, "r");
    if (!file) return 0;

    int count = 0;
    char line[4096];
    while (fgets(line, sizeof(line), file)) count++;
    rewind(file);

    *inputs = (long double**)malloc(count * sizeof(long double*));
    *targets = (long double**)malloc(count * sizeof(long double*));

    for (int i = 0; i < count; i++) {
        (*inputs)[i] = (long double*)malloc(input_dim * sizeof(long double));
        (*targets)[i] = (long double*)malloc(target_dim * sizeof(long double));
        fgets(line, sizeof(line), file);
        char *token = strtok(line, ",");
        for (int j = 0; j < input_dim && token; j++) {
            (*inputs)[i][j] = strtold(token, NULL);
            token = strtok(NULL, ",");
        }
        for (int j = 0; j < target_dim && token; j++) {
            (*targets)[i][j] = strtold(token, NULL);
            token = strtok(NULL, ",");
        }
    }
    fclose(file);
    return count;
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    if (argc < 2) {
        printf("Usage: %s <data.csv>\n", argv[0]);
        printf("Defaulting to internal XOR test...\n");
    }

    int layers[] = {2, 4, 1};
    DeepMLP *mlp = create_mlp(3, layers);

    long double **inputs, **targets;
    int num_samples;

    if (argc >= 2) {
        num_samples = load_csv(argv[1], &inputs, &targets, 2, 1);
        if (num_samples == 0) {
            printf("Error loading CSV file.\n");
            return 1;
        }
    } else {
        num_samples = 4;
        inputs = (long double**)malloc(4 * sizeof(long double*));
        targets = (long double**)malloc(4 * sizeof(long double*));
        long double raw_in[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
        long double raw_tar[4][1] = {{0}, {1}, {1}, {0}};
        for (int i = 0; i < 4; i++) {
            inputs[i] = (long double*)malloc(2 * sizeof(long double));
            targets[i] = (long double*)malloc(1 * sizeof(long double));
            memcpy(inputs[i], raw_in[i], 2 * sizeof(long double));
            memcpy(targets[i], raw_tar[i], 1 * sizeof(long double));
        }
    }

    printf("Starting Training with %d samples...\n", num_samples);
    for (int epoch = 0; epoch <= 100000; epoch++) {
        for (int i = 0; i < num_samples; i++) {
            forward_prop(mlp, inputs[i]);
            back_prop(mlp, targets[i], 0.1L);
        }
        if (epoch % 10000 == 0) {
            long double err = 0;
            for (int i = 0; i < num_samples; i++) {
                forward_prop(mlp, inputs[i]);
                err += fabsl(targets[i][0] - mlp->nodes[2][0]);
            }
            printf("Epoch %d, Avg Error: %.15Lf\n", epoch, err / num_samples);
        }
    }

    printf("\nTest Results:\n");
    for (int i = 0; i < num_samples; i++) {
        forward_prop(mlp, inputs[i]);
        printf("In: %.2Lf,%.2Lf Target: %.0Lf Pred: %.15Lf\n", 
               inputs[i][0], inputs[i][1], targets[i][0], mlp->nodes[2][0]);
    }

    return 0;
}
