#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* パラメータ */
#define SIGMA 10.0L
#define RHO   28.0L
#define BETA  (8.0L/3.0L)
#define HBAR  0.1L

typedef struct {
    int num_layers;
    int *layer_sizes;
    long double **nodes;
    long double ***weights;
    long double **biases;
    long double **deltas;
} DNN;

long double sigmoid_ld(long double x) {
    return 1.0L / (1.0L + expl(-x));
}

long double sigmoid_derivative_ld(long double x) {
    return x * (1.0L - x);
}

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
            for (int j = 0; j < layer_sizes[i-1]; j++) {
                dnn->weights[i-1][j] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
                for (int k = 0; k < layer_sizes[i]; k++) {
                    dnn->weights[i-1][j][k] = (((long double)rand() / RAND_MAX) * 2.0L - 1.0L) * sqrtl(2.0L / layer_sizes[i-1]);
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
            dnn->nodes[i][j] = sigmoid_ld(sum);
        }
    }
}

void back_prop(DNN *dnn, long double *targets, long double lr) {
    int last = dnn->num_layers - 1;
    for (int i = 0; i < dnn->layer_sizes[last]; i++) {
        long double error = targets[i] - dnn->nodes[last][i];
        dnn->deltas[last][i] = error * sigmoid_derivative_ld(dnn->nodes[last][i]);
    }
    for (int i = last - 1; i > 0; i--) {
        for (int j = 0; j < dnn->layer_sizes[i]; j++) {
            long double error = 0.0L;
            for (int k = 0; k < dnn->layer_sizes[i+1]; k++) {
                error += dnn->deltas[i+1][k] * dnn->weights[i][j][k];
            }
            dnn->deltas[i][j] = error * sigmoid_derivative_ld(dnn->nodes[i][j]);
        }
    }
    for (int i = 1; i < dnn->num_layers; i++) {
        for (int j = 0; j < dnn->layer_sizes[i]; j++) {
            dnn->biases[i][j] += lr * dnn->deltas[i][j];
            for (int k = 0; k < dnn->layer_sizes[i-1]; k++) {
                dnn->weights[i-1][k][j] += lr * dnn->deltas[i][j] * dnn->nodes[i-1][k];
            }
        }
    }
}

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

void rk4_step(void (*f)(long double, long double, long double, long double*, long double*, long double*),
              long double *x, long double *y, long double *z, long double dt) {
    long double k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
    f(*x, *y, *z, &k1x, &k1y, &k1z);
    f(*x + 0.5L*dt*k1x, *y + 0.5L*dt*k1y, *z + 0.5L*dt*k1z, &k2x, &k2y, &k2z);
    f(*x + 0.5L*dt*k2x, *y + 0.5L*dt*k2y, *z + 0.5L*dt*k2z, &k3x, &k3y, &k3z);
    f(*x + dt*k3x, *y + dt*k3y, *z + dt*k3z, &k4x, &k4y, &k4z);
    *x += (dt/6.0L)*(k1x + 2*k2x + 2*k3x + k4x);
    *y += (dt/6.0L)*(k1y + 2*k2y + 2*k3y + k4y);
    *z += (dt/6.0L)*(k1z + 2*k2z + 2*k3z + k4z);
}

int main() {
    srand(time(NULL));
    int layers[] = {3, 16, 16, 3};
    DNN *dnn = create_dnn(4, layers);

    long double xc = 1.0L, yc = 1.0L, zc = 1.0L;
    long double xq = 1.0L, yq = 1.0L, zq = 1.0L;
    long double dt = 0.01L;
    long double lr = 0.1L;

    printf("--- Deep Learning (DNN) Lorenz Quantization Comparison (long double) ---\n");
    printf("Training DNN to learn Quantum Correction...\n");

    for (int i = 0; i < 10000; i++) {
        long double inputs[3] = {xc / 50.0L, yc / 50.0L, zc / 50.0L}; // Normalization
        long double target[3] = {xq / 50.0L, yq / 50.0L, zq / 50.0L};

        forward_prop(dnn, inputs);
        back_prop(dnn, target, lr);

        rk4_step(lorenz_classical, &xc, &yc, &zc, dt);
        rk4_step(lorenz_semiclassical, &xq, &yq, &zq, dt);

        if (i % 1000 == 0) {
            long double diff = sqrtl(powl(xc-xq, 2) + powl(yc-yq, 2) + powl(zc-zq, 2));
            printf("Step %d: Classical-Quantum Diff = %.10Lf\n", i, diff);
        }
    }

    printf("\nComparison Result:\n");
    printf("Time, Classical(X), Quantum(X), DNN_Predicted(X), Error(Q-DNN)\n");
    
    xc = 1.0L; yc = 1.0L; zc = 1.0L;
    xq = 1.0L; yq = 1.0L; zq = 1.0L;
    for (int i = 0; i < 100; i++) {
        long double inputs[3] = {xc / 50.0L, yc / 50.0L, zc / 50.0L};
        forward_prop(dnn, inputs);
        long double dnn_x = dnn->nodes[3][0] * 50.0L;

        printf("%.2Lf, %.6Lf, %.6Lf, %.6Lf, %.6Lf\n", i*dt, xc, xq, dnn_x, fabsl(xq - dnn_x));

        rk4_step(lorenz_classical, &xc, &yc, &zc, dt);
        rk4_step(lorenz_semiclassical, &xq, &yq, &zq, dt);
    }

    return 0;
}
