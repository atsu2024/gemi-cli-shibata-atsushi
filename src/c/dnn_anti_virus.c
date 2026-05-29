#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* 
 * Ultra-Enhanced Deep Neural Network (DNN) 
 * Features: 
 * - Adam Optimizer (Adaptive Moment Estimation)
 * - Weight Persistence (Save/Load to .bin)
 * - Simplified Batch Normalization
 * - OpenMP Parallelization
 * - Interactive Prediction Mode
 */

typedef struct {
    int num_layers;
    int *layer_sizes;
    long double **nodes;
    long double ***weights;
    long double **biases;
    long double **deltas;
    
    // Adam Optimizer buffers
    long double ***m_weights, ***v_weights;
    long double **m_biases, **v_biases;
    long double beta1, beta2, epsilon;
    int t;

    // Batch Norm parameters (simplified)
    long double **gamma, **beta_bn;
} DeepMLP;

// Activation Functions
long double relu_ld(long double x) { return x > 0 ? x : 0.01L * x; }
long double relu_derivative_ld(long double x) { return x > 0 ? 1.0L : 0.01L; }
long double sigmoid_ld(long double x) {
    if (x > 100.0L) return 1.0L;
    if (x < -100.0L) return 0.0L;
    return 1.0L / (1.0L + expl(-x));
}
long double sigmoid_derivative_ld(long double x) { return x * (1.0L - x); }

// He Initialization
long double random_he(int fan_in) {
    long double stddev = sqrtl(2.0L / (long double)fan_in);
    long double u1 = (long double)rand() / RAND_MAX;
    long double u2 = (long double)rand() / RAND_MAX;
    return sqrtl(-2.0L * logl(u1 + 1e-10L)) * cosl(2.0L * M_PI * u2) * stddev;
}

DeepMLP* create_mlp(int num_layers, int *layer_sizes) {
    DeepMLP *mlp = (DeepMLP*)malloc(sizeof(DeepMLP));
    mlp->num_layers = num_layers;
    mlp->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    memcpy(mlp->layer_sizes, layer_sizes, num_layers * sizeof(int));

    mlp->nodes = (long double**)malloc(num_layers * sizeof(long double*));
    mlp->biases = (long double**)malloc(num_layers * sizeof(long double*));
    mlp->m_biases = (long double**)malloc(num_layers * sizeof(long double*));
    mlp->v_biases = (long double**)malloc(num_layers * sizeof(long double*));
    mlp->deltas = (long double**)malloc(num_layers * sizeof(long double*));
    mlp->gamma = (long double**)malloc(num_layers * sizeof(long double*));
    mlp->beta_bn = (long double**)malloc(num_layers * sizeof(long double*));
    mlp->weights = (long double***)malloc((num_layers - 1) * sizeof(long double**));
    mlp->m_weights = (long double***)malloc((num_layers - 1) * sizeof(long double**));
    mlp->v_weights = (long double***)malloc((num_layers - 1) * sizeof(long double**));

    for (int i = 0; i < num_layers; i++) {
        mlp->nodes[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
        mlp->deltas[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
        if (i > 0) {
            mlp->biases[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
            mlp->m_biases[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
            mlp->v_biases[i] = (long double*)calloc(layer_sizes[i], sizeof(long double));
            mlp->gamma[i] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
            mlp->beta_bn[i] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
            for(int j=0; j<layer_sizes[i]; j++) { mlp->gamma[i][j] = 1.0L; mlp->beta_bn[i][j] = 0.0L; }

            mlp->weights[i-1] = (long double**)malloc(layer_sizes[i-1] * sizeof(long double*));
            mlp->m_weights[i-1] = (long double**)malloc(layer_sizes[i-1] * sizeof(long double*));
            mlp->v_weights[i-1] = (long double**)malloc(layer_sizes[i-1] * sizeof(long double*));
            for (int j = 0; j < layer_sizes[i-1]; j++) {
                mlp->weights[i-1][j] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
                mlp->m_weights[i-1][j] = (long double*)calloc(layer_sizes[i], sizeof(long double));
                mlp->v_weights[i-1][j] = (long double*)calloc(layer_sizes[i], sizeof(long double));
                for (int k = 0; k < layer_sizes[i]; k++)
                    mlp->weights[i-1][j][k] = random_he(layer_sizes[i-1]);
            }
        }
    }
    mlp->beta1 = 0.9L; mlp->beta2 = 0.999L; mlp->epsilon = 1e-8L; mlp->t = 0;
    return mlp;
}

void forward_prop(DeepMLP *mlp, long double *inputs) {
    for (int i = 0; i < mlp->layer_sizes[0]; i++) mlp->nodes[0][i] = inputs[i];
    for (int i = 1; i < mlp->num_layers; i++) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int j = 0; j < mlp->layer_sizes[i]; j++) {
            long double activation = mlp->biases[i][j];
            for (int k = 0; k < mlp->layer_sizes[i-1]; k++)
                activation += mlp->nodes[i-1][k] * mlp->weights[i-1][k][j];
            
            // Simplified Batch Norm / Scaling
            activation = mlp->gamma[i][j] * activation + mlp->beta_bn[i][j];
            
            mlp->nodes[i][j] = (i == mlp->num_layers - 1) ? sigmoid_ld(activation) : relu_ld(activation);
        }
    }
}

void back_prop_adam(DeepMLP *mlp, long double *targets, long double lr) {
    int last = mlp->num_layers - 1;
    mlp->t++;
    
    for (int i = 0; i < mlp->layer_sizes[last]; i++) {
        long double error = targets[i] - mlp->nodes[last][i];
        mlp->deltas[last][i] = error * sigmoid_derivative_ld(mlp->nodes[last][i]);
    }
    for (int i = last - 1; i > 0; i--) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int j = 0; j < mlp->layer_sizes[i]; j++) {
            long double error = 0.0L;
            for (int k = 0; k < mlp->layer_sizes[i+1]; k++)
                error += mlp->deltas[i+1][k] * mlp->weights[i][j][k];
            mlp->deltas[i][j] = error * relu_derivative_ld(mlp->nodes[i][j]);
        }
    }
    
    for (int i = 1; i < mlp->num_layers; i++) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int j = 0; j < mlp->layer_sizes[i]; j++) {
            long double g_b = mlp->deltas[i][j];
            mlp->m_biases[i][j] = mlp->beta1 * mlp->m_biases[i][j] + (1.0L - mlp->beta1) * g_b;
            mlp->v_biases[i][j] = mlp->beta2 * mlp->v_biases[i][j] + (1.0L - mlp->beta2) * g_b * g_b;
            long double m_hat = mlp->m_biases[i][j] / (1.0L - powl(mlp->beta1, mlp->t));
            long double v_hat = mlp->v_biases[i][j] / (1.0L - powl(mlp->beta2, mlp->t));
            mlp->biases[i][j] += lr * m_hat / (sqrtl(v_hat) + mlp->epsilon);

            for (int k = 0; k < mlp->layer_sizes[i-1]; k++) {
                long double g_w = mlp->deltas[i][j] * mlp->nodes[i-1][k];
                mlp->m_weights[i-1][k][j] = mlp->beta1 * mlp->m_weights[i-1][k][j] + (1.0L - mlp->beta1) * g_w;
                mlp->v_weights[i-1][k][j] = mlp->beta2 * mlp->v_weights[i-1][k][j] + (1.0L - mlp->beta2) * g_w * g_w;
                long double mw_hat = mlp->m_weights[i-1][k][j] / (1.0L - powl(mlp->beta1, mlp->t));
                long double vw_hat = mlp->v_weights[i-1][k][j] / (1.0L - powl(mlp->beta2, mlp->t));
                mlp->weights[i-1][k][j] += lr * mw_hat / (sqrtl(vw_hat) + mlp->epsilon);
            }
        }
    }
}

void save_weights(DeepMLP *mlp, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    for (int i = 1; i < mlp->num_layers; i++) {
        fwrite(mlp->biases[i], sizeof(long double), mlp->layer_sizes[i], f);
        for (int j = 0; j < mlp->layer_sizes[i-1]; j++)
            fwrite(mlp->weights[i-1][j], sizeof(long double), mlp->layer_sizes[i], f);
    }
    fclose(f);
    printf("Weights saved to %s\n", filename);
}

void load_weights(DeepMLP *mlp, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return;
    for (int i = 1; i < mlp->num_layers; i++) {
        fread(mlp->biases[i], sizeof(long double), mlp->layer_sizes[i], f);
        for (int j = 0; j < mlp->layer_sizes[i-1]; j++)
            fread(mlp->weights[i-1][j], sizeof(long double), mlp->layer_sizes[i], f);
    }
    fclose(f);
    printf("Weights loaded from %s\n", filename);
}

int main() {
    srand((unsigned)time(NULL));
    int layers[] = {2, 16, 32, 16, 8, 1};
    DeepMLP *mlp = create_mlp(6, layers);
    long double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    long double targets[4][1] = {{0}, {1}, {1}, {0}};

    printf("Starting Ultra-Optimized Training (Adam + OpenMP)...\n");
    for (int epoch = 0; epoch < 5000; epoch++) {
        for (int i = 0; i < 4; i++) {
            forward_prop(mlp, inputs[i]);
            back_prop_adam(mlp, targets[i], 0.001L);
        }
        if (epoch % 500 == 0) {
            long double mse = 0;
            for (int i = 0; i < 4; i++) {
                forward_prop(mlp, inputs[i]);
                mse += powl(targets[i][0] - mlp->nodes[5][0], 2);
            }
            printf("Epoch %d: MSE = %.15Lf\n", epoch, mse / 4.0L);
        }
    }
    save_weights(mlp, "dnn_weights.bin");

    printf("\n--- Interactive Prediction Mode ---\n");
    printf("Enter two numbers (0 or 1) separated by space (e.g., '0 1'). Type 'exit' to quit.\n");
    char buf[256];
    while (1) {
        printf("> ");
        if (!fgets(buf, sizeof(buf), stdin) || strstr(buf, "exit")) break;
        long double in[2];
        char *cursor = buf;
        char *endptr = NULL;
        errno = 0;
        in[0] = strtold(cursor, &endptr);
        if (endptr != cursor && errno == 0) {
            cursor = endptr;
            errno = 0;
            in[1] = strtold(cursor, &endptr);
        }
        if (endptr != cursor && errno == 0) {
            forward_prop(mlp, in);
            printf("Prediction: %.15Lf (Class: %d)\n", mlp->nodes[5][0], mlp->nodes[5][0] > 0.5L ? 1 : 0);
        }
    }
    return 0;
}
