#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "libdnn.h"

#define SIGMA 10.0L
#define RHO   28.0L
#define BETA  (8.0L/3.0L)
#define HBAR  0.1L
#define DT    0.01L
#define LEARN_RATE 0.01L
#define EPOCHS     100000
#define NORM_FACTOR 50.0L
#define WEIGHT_FILE "data/lorenz_dnn_weights.bin"
#define OUTPUT_FILE "data/lorenz_final_results.csv"

void lorenz_semiclassical(long double x, long double y, long double z, long double *dx, long double *dy, long double *dz) {
    long double qc = (HBAR * HBAR) / 12.0L;
    *dx = SIGMA * (y - x); *dy = x * (RHO - z) - y - qc; *dz = x * y - BETA * z;
}

void rk4_step(long double *x, long double *y, long double *z, long double dt) {
    long double k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
    lorenz_semiclassical(*x, *y, *z, &k1x, &k1y, &k1z);
    lorenz_semiclassical(*x + 0.5L*dt*k1x, *y + 0.5L*dt*k1y, *z + 0.5L*dt*k1z, &k2x, &k2y, &k2z);
    lorenz_semiclassical(*x + 0.5L*dt*k2x, *y + 0.5L*dt*k2y, *z + 0.5L*dt*k2z, &k3x, &k3y, &k3z);
    lorenz_semiclassical(*x + dt*k3x, *y + dt*k3y, *z + dt*k3z, &k4x, &k4y, &k4z);
    *x += (dt/6.0L)*(k1x + 2*k2x + 2*k3x + k4x); *y += (dt/6.0L)*(k1y + 2*k2y + 2*k3y + k4y); *z += (dt/6.0L)*(k1z + 2*k2z + 2*k3z + k4z);
}

int main() {
    srand(time(NULL));
    int layers[] = {3, 128, 128, 64, 3};
    DNN *dnn = create_dnn(5, layers);

    if (load_weights(dnn, WEIGHT_FILE)) {
        printf("Pre-trained weights loaded from %s.\n", WEIGHT_FILE);
    } else {
        printf("Training DNN (this may take a moment)...\n");
        long double x = 1.0L, y = 1.0L, z = 1.0L;
        for (int i = 0; i < EPOCHS; i++) {
            long double current[3] = {x / NORM_FACTOR, y / NORM_FACTOR, z / NORM_FACTOR};
            long double next_x = x, next_y = y, next_z = z;
            rk4_step(&next_x, &next_y, &next_z, DT);
            long double target[3] = {next_x / NORM_FACTOR, next_y / NORM_FACTOR, next_z / NORM_FACTOR};
            forward_prop(dnn, current);
            back_prop(dnn, target, LEARN_RATE);
            x = next_x; y = next_y; z = next_z;
            if (fabsl(x) > 100.0L) { x=1.0L; y=1.0L; z=1.0L; }
            if (i % 10000 == 0) printf("Epoch %d/100000 complete.\n", i);
        }
        save_weights(dnn, WEIGHT_FILE);
    }

    FILE *csv = fopen(OUTPUT_FILE, "w");
    if (!csv) {
        perror("Error opening output file");
        free_dnn(dnn);
        return 1;
    }
    fprintf(csv, "Time,Actual_X,Actual_Y,Actual_Z,Pred_X,Pred_Y,Pred_Z\n");
    long double x = 2.0L, y = 2.0L, z = 2.0L;
    for (int i = 0; i < 1000; i++) {
        long double current[3] = {x / NORM_FACTOR, y / NORM_FACTOR, z / NORM_FACTOR};
        forward_prop(dnn, current);
        fprintf(csv, "%.3Lf,%.6Lf,%.6Lf,%.6Lf,%.6Lf,%.6Lf,%.6Lf\n", i*DT, x, y, z, dnn->nodes[dnn->num_layers-1][0]*NORM_FACTOR, dnn->nodes[dnn->num_layers-1][1]*NORM_FACTOR, dnn->nodes[dnn->num_layers-1][2]*NORM_FACTOR);
        rk4_step(&x, &y, &z, DT);
    }
    fclose(csv);
    printf("Processing complete. Data saved to %s\n", OUTPUT_FILE);
    
    free_dnn(dnn);
    return 0;
}
