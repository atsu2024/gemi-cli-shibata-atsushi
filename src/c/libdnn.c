#include "libdnn.h"

long double activation_ld(long double x) { return tanhl(x); }
long double activation_derivative_ld(long double x) { return 1.0L - x * x; }

DNN* create_dnn(int num_layers, int *layer_sizes) {
    DNN *dnn = (DNN*)malloc(sizeof(DNN));
    if (!dnn) return NULL;
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
                for (int k = 0; k < layer_sizes[i]; k++)
                    dnn->weights[i-1][j][k] = (((long double)rand() / RAND_MAX) * 2.0L - 1.0L) * sqrtl(6.0L / (layer_sizes[i-1] + layer_sizes[i]));
            }
        } else {
            dnn->biases[i] = NULL; // No biases for input layer
        }
    }
    return dnn;
}

void free_dnn(DNN *dnn) {
    if (!dnn) return;
    for (int i = 0; i < dnn->num_layers; i++) {
        free(dnn->nodes[i]);
        free(dnn->deltas[i]);
        if (i > 0) {
            free(dnn->biases[i]);
            for (int j = 0; j < dnn->layer_sizes[i-1]; j++) {
                free(dnn->weights[i-1][j]);
            }
            free(dnn->weights[i-1]);
        }
    }
    free(dnn->nodes);
    free(dnn->deltas);
    free(dnn->biases);
    free(dnn->weights);
    free(dnn->layer_sizes);
    free(dnn);
}

void save_weights(DNN *dnn, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return;
    fwrite(&dnn->num_layers, sizeof(int), 1, fp);
    fwrite(dnn->layer_sizes, sizeof(int), dnn->num_layers, fp);
    for (int i = 1; i < dnn->num_layers; i++) {
        fwrite(dnn->biases[i], sizeof(long double), dnn->layer_sizes[i], fp);
        for (int j = 0; j < dnn->layer_sizes[i-1]; j++)
            fwrite(dnn->weights[i-1][j], sizeof(long double), dnn->layer_sizes[i], fp);
    }
    fclose(fp);
}

int load_weights(DNN *dnn, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 0;
    int nl;
    if (fread(&nl, sizeof(int), 1, fp) != 1) { fclose(fp); return 0; }
    if (nl != dnn->num_layers) { fclose(fp); return 0; }
    
    int *ls = (int*)malloc(nl * sizeof(int));
    if (fread(ls, sizeof(int), nl, fp) != (size_t)nl) { free(ls); fclose(fp); return 0; }
    
    for (int i = 1; i < dnn->num_layers; i++) {
        fread(dnn->biases[i], sizeof(long double), dnn->layer_sizes[i], fp);
        for (int j = 0; j < dnn->layer_sizes[i-1]; j++)
            fread(dnn->weights[i-1][j], sizeof(long double), dnn->layer_sizes[i], fp);
    }
    free(ls);
    fclose(fp);
    return 1;
}

void forward_prop(DNN *dnn, long double *inputs) {
    for (int i = 0; i < dnn->layer_sizes[0]; i++) dnn->nodes[0][i] = inputs[i];
    for (int i = 1; i < dnn->num_layers; i++) {
        for (int j = 0; j < dnn->layer_sizes[i]; j++) {
            long double sum = dnn->biases[i][j];
            for (int k = 0; k < dnn->layer_sizes[i-1]; k++) sum += dnn->nodes[i-1][k] * dnn->weights[i-1][k][j];
            dnn->nodes[i][j] = activation_ld(sum);
        }
    }
}

void back_prop(DNN *dnn, long double *targets, long double lr) {
    int last = dnn->num_layers - 1;
    for (int i = 0; i < dnn->layer_sizes[last]; i++)
        dnn->deltas[last][i] = (targets[i] - dnn->nodes[last][i]) * activation_derivative_ld(dnn->nodes[last][i]);
    for (int i = last - 1; i > 0; i--) {
        for (int j = 0; j < dnn->layer_sizes[i]; j++) {
            long double error = 0.0L;
            for (int k = 0; k < dnn->layer_sizes[i+1]; k++) error += dnn->deltas[i+1][k] * dnn->weights[i][j][k];
            dnn->deltas[i][j] = error * activation_derivative_ld(dnn->nodes[i][j]);
        }
    }
    for (int i = 1; i < dnn->num_layers; i++) {
        for (int j = 0; j < dnn->layer_sizes[i]; j++) {
            dnn->biases[i][j] += lr * dnn->deltas[i][j];
            for (int k = 0; k < dnn->layer_sizes[i-1]; k++) dnn->weights[i-1][k][j] += lr * dnn->deltas[i][j] * dnn->nodes[i-1][k];
        }
    }
}
