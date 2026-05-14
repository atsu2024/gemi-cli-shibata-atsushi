#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/*
 * dnn_002_pdf_ld.c
 * Deep Neural Network (Autoencoder) for processing image data related to 002.pdf.
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
    srand((unsigned)time(NULL));
    
    // Load image data (associated with 002.pdf)
    const int input_size = 1024;
    long double *input_data = (long double*)malloc(input_size * sizeof(long double));
    FILE *fp = fopen("image_data.csv", "r");
    if (!fp) {
        printf("Error: image_data.csv not found. Creating synthetic data...\n");
        for (int i = 0; i < input_size; i++) input_data[i] = (long double)rand() / RAND_MAX;
    } else {
        char line[128];
        for (int i = 0; i < input_size; i++) {
            if (fgets(line, sizeof(line), fp) == NULL) break;
            input_data[i] = strtold(line, NULL);
            if (isnan(input_data[i]) || isinf(input_data[i])) {
                input_data[i] = 0.0L;
            }
        }
        fclose(fp);
        printf("Loaded 1024 points from image_data.csv. Sample[0]: %Lf, Sample[512]: %Lf, Sample[1023]: %Lf\n", 
               input_data[0], input_data[512], input_data[1023]);
    }

    int layers[] = {1024, 512, 256, 512, 1024};
    DeepMLP *mlp = create_mlp(5, layers);
    
    // Check if weights are initialized correctly
    printf("Deep Network Initialized with 5 layers.\n");
    printf("Weight Sample[0][0][0]: %Lf\n", mlp->weights[0][0][0]);

    long double lr = 0.000001L;
    int epochs = 2000;

    printf("\nStarting training for 002.pdf Deep Learning project...\n");
    for (int e = 0; e <= epochs; e++) {
        forward_prop(mlp, input_data);
        back_prop(mlp, input_data, lr); // Autoencoder: target is input
        
        if (e % 100 == 0) {
            long double mse = 0;
            int last_layer = mlp->num_layers - 1;
            for (int i = 0; i < input_size; i++) {
                mse += powl(input_data[i] - mlp->nodes[last_layer][i], 2.0L);
            }
            printf("Epoch %d: MSE = %.20Lf\n", e, mse / (long double)input_size);
        }
    }

    printf("\nSaving reconstructed data to output_002.csv...\n");
    fp = fopen("output_002.csv", "w");
    if (fp) {
        int last_layer = mlp->num_layers - 1;
        for (int i = 0; i < input_size; i++) {
            fprintf(fp, "%.20Lf\n", mlp->nodes[last_layer][i]);
        }
        fclose(fp);
        printf("Reconstructed data saved successfully.\n");
    }

    return 0;
}
