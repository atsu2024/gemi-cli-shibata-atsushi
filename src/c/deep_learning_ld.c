#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

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
    
    FILE *fp = fopen("xor_data.csv", "r");
    if (!fp) { printf("Error: xor_data.csv not found.\n"); return 1; }

    long double data[4][3];
    char line[1024];
    int count = 0;
    while (fgets(line, sizeof(line), fp) && count < 4) {
        char *ptr;
        data[count][0] = strtold(line, &ptr);
        if (*ptr == ',') ptr++;
        data[count][1] = strtold(ptr, &ptr);
        if (*ptr == ',') ptr++;
        data[count][2] = strtold(ptr, NULL);
        printf("Loaded Data [%d]: %Lf, %Lf -> %Lf\n", count, data[count][0], data[count][1], data[count][2]);
        count++;
    }
    fclose(fp);

    int layers[] = {2, 4, 1};
    DeepMLP *mlp = create_mlp(3, layers);

    long double lr = 0.1L; // 学習率を下げて安定化
    int epochs = 100000;

    printf("\nStarting training...\n");
    for (int e = 0; e <= epochs; e++) {
        long double total_error = 0;
        for (int i = 0; i < 4; i++) {
            long double inputs[2] = {data[i][0], data[i][1]};
            long double targets[1] = {data[i][2]};
            forward_prop(mlp, inputs);
            back_prop(mlp, targets, lr);
            total_error += powl(targets[0] - mlp->nodes[2][0], 2.0L);
        }
        if (e % 10000 == 0) {
            printf("Epoch %d: MSE = %.20Lf\n", e, total_error / 4.0L);
        }
    }

    printf("\nTest Results:\n");
    for (int i = 0; i < 4; i++) {
        long double inputs[2] = {data[i][0], data[i][1]};
        forward_prop(mlp, inputs);
        printf("In: %Lf, %Lf -> Out: %.20Lf (Expected: %Lf)\n", 
               inputs[0], inputs[1], mlp->nodes[2][0], data[count-4+i][2]);
    }

    return 0;
}
