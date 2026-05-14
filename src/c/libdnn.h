#ifndef LIBDNN_H
#define LIBDNN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    int num_layers;
    int *layer_sizes;
    long double **nodes;
    long double ***weights;
    long double **biases;
    long double **deltas;
} DNN;

// Activation functions
long double activation_ld(long double x);
long double activation_derivative_ld(long double x);

// Lifecycle management
DNN* create_dnn(int num_layers, int *layer_sizes);
void free_dnn(DNN *dnn);

// Persistence
void save_weights(DNN *dnn, const char *filename);
int load_weights(DNN *dnn, const char *filename);

// Core operations
void forward_prop(DNN *dnn, long double *inputs);
void back_prop(DNN *dnn, long double *targets, long double lr);

#endif // LIBDNN_H
