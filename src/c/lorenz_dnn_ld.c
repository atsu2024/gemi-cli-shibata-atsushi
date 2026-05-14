#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* Lorenz Attractor Parameters */
#define SIGMA 10.0L
#define RHO   28.0L
#define BETA  (8.0L/3.0L)
#define DT    0.01L

/* DNN Parameters */
#define LEARN_RATE 0.05L
#define EPOCHS     20000
#define NORM_FACTOR 50.0L

typedef struct {
    int num_layers;
    int *layer_sizes;
    long double **nodes;
    long double ***weights;
    long double **biases;
    long double **deltas;
} DNN;

/* Activation function: Tanh for better range in regression */
long double activation_ld(long double x) {
    return tanhl(x);
}

long double activation_derivative_ld(long double x) {
    return 1.0L - x * x;
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
            dnn->biases[i] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
            dnn->weights[i-1] = (long double**)malloc(layer_sizes[i-1] * sizeof(long double*));
            for (int j = 0; j < layer_sizes[i-1]; j++) {
                dnn->weights[i-1][j] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
                for (int k = 0; k < layer_sizes[i]; k++) {
                    // Xavier/Glorot initialization
                    dnn->weights[i-1][j][k] = (((long double)rand() / RAND_MAX) * 2.0L - 1.0L) * sqrtl(6.0L / (layer_sizes[i-1] + layer_sizes[i]));
                }
            }
            for (int k = 0; k < layer_sizes[i]; k++) {
                dnn->biases[i][k] = 0.0L;
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

void lorenz_deriv(long double x, long double y, long double z, long double *dx, long double *dy, long double *dz) {
    *dx = SIGMA * (y - x);
    *dy = x * (RHO - z) - y;
    *dz = x * y - BETA * z;
}

void rk4_step(long double *x, long double *y, long double *z, long double dt) {
    long double k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
    lorenz_deriv(*x, *y, *z, &k1x, &k1y, &k1z);
    lorenz_deriv(*x + 0.5L*dt*k1x, *y + 0.5L*dt*k1y, *z + 0.5L*dt*k1z, &k2x, &k2y, &k2z);
    lorenz_deriv(*x + 0.5L*dt*k2x, *y + 0.5L*dt*k2y, *z + 0.5L*dt*k2z, &k3x, &k3y, &k3z);
    lorenz_deriv(*x + dt*k3x, *y + dt*k3y, *z + dt*k3z, &k4x, &k4y, &k4z);
    *x += (dt/6.0L)*(k1x + 2*k2x + 2*k3x + k4x);
    *y += (dt/6.0L)*(k1y + 2*k2y + 2*k3y + k4y);
    *z += (dt/6.0L)*(k1z + 2*k2z + 2*k3z + k4z);
}

int main() {
    srand(time(NULL));
    int layers[] = {3, 64, 64, 3};
    DNN *dnn = create_dnn(4, layers);

    long double x = 1.0L, y = 1.0L, z = 1.0L;
    
    printf("--- Lorenz Attractor Deep Learning (DNN) with long double ---\n");
    printf("Training DNN to predict next state (x,y,z) at t+dt...\n");

    for (int i = 0; i < EPOCHS; i++) {
        long double current[3] = {x / NORM_FACTOR, y / NORM_FACTOR, z / NORM_FACTOR};
        
        // Calculate next state using RK4
        long double next_x = x, next_y = y, next_z = z;
        rk4_step(&next_x, &next_y, &next_z, DT);
        
        long double target[3] = {next_x / NORM_FACTOR, next_y / NORM_FACTOR, next_z / NORM_FACTOR};

        // Train DNN
        forward_prop(dnn, current);
        back_prop(dnn, target, LEARN_RATE);

        // Move to next state for next training sample
        x = next_x; y = next_y; z = next_z;

        // If it goes out of bounds, reset
        if (fabsl(x) > 100.0L || fabsl(y) > 100.0L || fabsl(z) > 100.0L) {
            x = 1.0L; y = 1.0L; z = 1.0L;
        }

        if (i % 2000 == 0) {
            long double loss = 0;
            for(int j=0; j<3; j++) loss += powl(target[j] - dnn->nodes[3][j], 2);
            printf("Epoch %d: Loss = %.10Lf\n", i, loss/3.0L);
        }
    }

    printf("\nTesting Prediction:\n");
    printf("Time, Actual(X), Predicted(X), Error\n");
    
    x = 2.0L; y = 2.0L; z = 2.0L; // New initial condition
    for (int i = 0; i < 20; i++) {
        long double inputs[3] = {x / NORM_FACTOR, y / NORM_FACTOR, z / NORM_FACTOR};
        forward_prop(dnn, inputs);
        long double pred_x = dnn->nodes[3][0] * NORM_FACTOR;

        long double actual_x = x;
        rk4_step(&actual_x, &y, &z, DT); // actual_x becomes next_x

        printf("%6.2Lf, %10.6Lf, %10.6Lf, %10.6Lf\n", i*DT, actual_x, pred_x, fabsl(actual_x - pred_x));
        x = actual_x;
    }

    return 0;
}
