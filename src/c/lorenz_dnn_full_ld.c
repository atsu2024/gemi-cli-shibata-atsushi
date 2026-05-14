#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* Lorenz Attractor Parameters (Classical & Semiclassical) */
#define SIGMA 10.0L
#define RHO   28.0L
#define BETA  (8.0L/3.0L)
#define HBAR  0.1L
#define DT    0.01L

/* DNN Parameters */
#define LEARN_RATE 0.02L
#define EPOCHS     50000
#define NORM_FACTOR 50.0L

typedef struct {
    int num_layers;
    int *layer_sizes;
    long double **nodes;
    long double ***weights;
    long double **biases;
    long double **deltas;
} DNN;

/* Activation function: Tanh for regression */
long double activation_ld(long double x) { return tanhl(x); }
long double activation_derivative_ld(long double x) { return 1.0L - x * x; }

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
            dnn->biases[i] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
            dnn->weights[i-1] = (long double**)malloc(layer_sizes[i-1] * sizeof(long double*));
            for (int j = 0; j < layer_sizes[i-1]; j++) {
                dnn->weights[i-1][j] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
                for (int k = 0; k < layer_sizes[i]; k++) {
                    dnn->weights[i-1][j][k] = (((long double)rand() / RAND_MAX) * 2.0L - 1.0L) * sqrtl(6.0L / (layer_sizes[i-1] + layer_sizes[i]));
                }
            }
            for (int k = 0; k < layer_sizes[i]; k++) dnn->biases[i][k] = 0.0L;
        }
    }
    return dnn;
}

void save_weights(DNN *dnn, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return;
    fwrite(&dnn->num_layers, sizeof(int), 1, fp);
    fwrite(dnn->layer_sizes, sizeof(int), dnn->num_layers, fp);
    for (int i = 1; i < dnn->num_layers; i++) {
        fwrite(dnn->biases[i], sizeof(long double), dnn->layer_sizes[i], fp);
        for (int j = 0; j < dnn->layer_sizes[i-1]; j++) {
            fwrite(dnn->weights[i-1][j], sizeof(long double), dnn->layer_sizes[i], fp);
        }
    }
    fclose(fp);
    printf("Weights saved to %s\n", filename);
}

void forward_prop(DNN *dnn, long double *inputs) {
    for (int i = 0; i < dnn->layer_sizes[0]; i++) dnn->nodes[0][i] = inputs[i];
    for (int i = 1; i < dnn->num_layers; i++) {
        for (int j = 0; j < dnn->layer_sizes[i]; j++) {
            long double sum = dnn->biases[i][j];
            for (int k = 0; k < dnn->layer_sizes[i-1]; k++) {
                sum += dnn->nodes[i-1][k] * dnn->weights[i-1][k][j];
            }
            dnn->nodes[i][j] = activation_ld(sum);
        }
    }
}

void back_prop(DNN *dnn, long double *targets, long double lr) {
    int last = dnn->num_layers - 1;
    for (int i = 0; i < dnn->layer_sizes[last]; i++) {
        long double error = targets[i] - dnn->nodes[last][i];
        dnn->deltas[last][i] = error * activation_derivative_ld(dnn->nodes[last][i]);
    }
    for (int i = last - 1; i > 0; i--) {
        for (int j = 0; j < dnn->layer_sizes[i]; j++) {
            long double error = 0.0L;
            for (int k = 0; k < dnn->layer_sizes[i+1]; k++) {
                error += dnn->deltas[i+1][k] * dnn->weights[i][j][k];
            }
            dnn->deltas[i][j] = error * activation_derivative_ld(dnn->nodes[i][j]);
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

/* Semiclassical Lorenz (Quantum Correction) */
void lorenz_semiclassical(long double x, long double y, long double z, long double *dx, long double *dy, long double *dz) {
    long double qc = (HBAR * HBAR) / 12.0L;
    *dx = SIGMA * (y - x);
    *dy = x * (RHO - z) - y - qc;
    *dz = x * y - BETA * z;
}

void rk4_step(long double *x, long double *y, long double *z, long double dt) {
    long double k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
    lorenz_semiclassical(*x, *y, *z, &k1x, &k1y, &k1z);
    lorenz_semiclassical(*x + 0.5L*dt*k1x, *y + 0.5L*dt*k1y, *z + 0.5L*dt*k1z, &k2x, &k2y, &k2z);
    lorenz_semiclassical(*x + 0.5L*dt*k2x, *y + 0.5L*dt*k2y, *z + 0.5L*dt*k2z, &k3x, &k3y, &k3z);
    lorenz_semiclassical(*x + dt*k3x, *y + dt*k3y, *z + dt*k3z, &k4x, &k4y, &k4z);
    *x += (dt/6.0L)*(k1x + 2*k2x + 2*k3x + k4x);
    *y += (dt/6.0L)*(k1y + 2*k2y + 2*k3y + k4y);
    *z += (dt/6.0L)*(k1z + 2*k2z + 2*k3z + k4z);
}

int main() {
    srand(time(NULL));
    // Complex Architecture: 3 (Input) -> 128 -> 128 -> 64 -> 3 (Output)
    int layers[] = {3, 128, 128, 64, 3};
    DNN *dnn = create_dnn(5, layers);

    long double x = 1.0L, y = 1.0L, z = 1.0L;
    printf("--- Full Lorenz DNN (Quantum Correction & Long Double) ---\n");
    printf("Architecture: 3-128-128-64-3\nTraining...\n");

    for (int i = 0; i < EPOCHS; i++) {
        long double current[3] = {x / NORM_FACTOR, y / NORM_FACTOR, z / NORM_FACTOR};
        long double next_x = x, next_y = y, next_z = z;
        rk4_step(&next_x, &next_y, &next_z, DT);
        long double target[3] = {next_x / NORM_FACTOR, next_y / NORM_FACTOR, next_z / NORM_FACTOR};

        forward_prop(dnn, current);
        back_prop(dnn, target, LEARN_RATE);

        x = next_x; y = next_y; z = next_z;
        if (fabsl(x) > 100.0L || fabsl(y) > 100.0L || fabsl(z) > 100.0L) { x = 1.0L; y = 1.0L; z = 1.0L; }

        if (i % 5000 == 0) {
            long double loss = 0;
            for(int j=0; j<3; j++) loss += powl(target[j] - dnn->nodes[4][j], 2);
            printf("Epoch %d: Loss = %.10Lf\n", i, loss/3.0L);
        }
    }

    save_weights(dnn, "lorenz_dnn_weights.bin");

    // Output to CSV for visualization
    FILE *csv = fopen("lorenz_comparison.csv", "w");
    fprintf(csv, "Time,Actual_X,Actual_Y,Actual_Z,Pred_X,Pred_Y,Pred_Z\n");
    
    x = 1.5L; y = 1.5L; z = 1.5L; // Validation condition
    for (int i = 0; i < 500; i++) {
        long double inputs[3] = {x / NORM_FACTOR, y / NORM_FACTOR, z / NORM_FACTOR};
        forward_prop(dnn, inputs);
        
        long double pred_x = dnn->nodes[4][0] * NORM_FACTOR;
        long double pred_y = dnn->nodes[4][1] * NORM_FACTOR;
        long double pred_z = dnn->nodes[4][2] * NORM_FACTOR;

        fprintf(csv, "%.3Lf,%.6Lf,%.6Lf,%.6Lf,%.6Lf,%.6Lf,%.6Lf\n", i*DT, x, y, z, pred_x, pred_y, pred_z);

        rk4_step(&x, &y, &z, DT);
    }
    fclose(csv);
    printf("Visualization data saved to lorenz_comparison.csv\n");

    return 0;
}
