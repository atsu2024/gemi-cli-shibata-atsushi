#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ローレンツパラメータ */
#define SIGMA 10.0L
#define RHO   28.0L
#define BETA  (8.0L/3.0L)
#define HBAR  0.1L  /* 量子補正係数 */

/* 学習パラメータ */
#define EPOCHS 20000
#define LR 0.01L

typedef struct {
    int layers;
    int *size;
    long double **n;   /* ノード */
    long double ***w;  /* 重み */
    long double **b;   /* バイアス */
} DNN;

/* 高精度活性化関数 (Tanh) */
long double active(long double x) { return tanhl(x); }
long double active_deriv(long double x) { return 1.0L - x * x; }

DNN* init_dnn(int layers, int *size) {
    DNN *d = malloc(sizeof(DNN));
    d->layers = layers;
    d->size = malloc(sizeof(int) * layers);
    for(int i=0; i<layers; i++) d->size[i] = size[i];
    
    d->n = malloc(sizeof(long double*) * layers);
    d->b = malloc(sizeof(long double*) * layers);
    d->w = malloc(sizeof(long double**) * (layers - 1));

    for (int i = 0; i < layers; i++) {
        d->n[i] = calloc(size[i], sizeof(long double));
        if (i > 0) {
            d->b[i] = calloc(size[i], sizeof(long double));
            d->w[i-1] = malloc(sizeof(long double*) * size[i-1]);
            for (int j = 0; j < size[i-1]; j++) {
                d->w[i-1][j] = malloc(sizeof(long double) * size[i]);
                for (int k = 0; k < size[i]; k++)
                    d->w[i-1][j][k] = ((long double)rand()/RAND_MAX - 0.5L) * sqrtl(2.0L/size[i-1]);
            }
        }
    }
    return d;
}

void forward(DNN *d, long double *in) {
    for (int j = 0; j < d->size[0]; j++) d->n[0][j] = in[j];
    for (int i = 1; i < d->layers; i++) {
        for (int j = 0; j < d->size[i]; j++) {
            long double sum = d->b[i][j];
            for (int k = 0; k < d->size[i-1]; k++)
                sum += d->n[i-1][k] * d->w[i-1][k][j];
            d->n[i][j] = (i == d->layers - 1) ? sum : active(sum);
        }
    }
}

void train(DNN *d, long double *target) {
    long double **delta = malloc(sizeof(long double*) * d->layers);
    for(int i=0; i<d->layers; i++) delta[i] = calloc(d->size[i], sizeof(long double));

    int last = d->layers - 1;
    for (int j = 0; j < d->size[last]; j++)
        delta[last][j] = target[j] - d->n[last][j];

    for (int i = last - 1; i > 0; i--) {
        for (int j = 0; j < d->size[i]; j++) {
            long double err = 0;
            for (int k = 0; k < d->size[i+1]; k++)
                err += delta[i+1][k] * d->w[i][j][k];
            delta[i][j] = err * active_deriv(d->n[i][j]);
        }
    }

    for (int i = 1; i < d->layers; i++) {
        for (int j = 0; j < d->size[i]; j++) {
            d->b[i][j] += LR * delta[i][j];
            for (int k = 0; k < d->size[i-1]; k++)
                d->w[i-1][k][j] += LR * delta[i][j] * d->n[i-1][k];
        }
    }
    for(int i=0; i<d->layers; i++) free(delta[i]);
    free(delta);
}

void lorenz_c(long double x, long double y, long double z, long double *dx, long double *dy, long double *dz) {
    *dx = SIGMA * (y - x); *dy = x * (RHO - z) - y; *dz = x * y - BETA * z;
}
void lorenz_q(long double x, long double y, long double z, long double *dx, long double *dy, long double *dz) {
    long double qc = (HBAR*HBAR)/12.0L;
    *dx = SIGMA * (y - x); *dy = x * (RHO - z) - y - qc; *dz = x * y - BETA * z;
}

void rk4(void (*f)(long double,long double,long double,long double*,long double*,long double*), 
         long double *x, long double *y, long double *z, long double dt) {
    long double k1x,k1y,k1z, k2x,k2y,k2z, k3x,k3y,k3z, k4x,k4y,k4z;
    f(*x,*y,*z,&k1x,&k1y,&k1z);
    f(*x+0.5L*dt*k1x, *y+0.5L*dt*k1y, *z+0.5L*dt*k1z, &k2x,&k2y,&k2z);
    f(*x+0.5L*dt*k2x, *y+0.5L*dt*k2y, *z+0.5L*dt*k2z, &k3x,&k3y,&k3z);
    f(*x+dt*k3x, *y+dt*k3y, *z+dt*k3z, &k4x,&k4y,&k4z);
    *x += (dt/6.0L)*(k1x+2*k2x+2*k3x+k4x); *y += (dt/6.0L)*(k1y+2*k2y+2*k3y+k4y); *z += (dt/6.0L)*(k1z+2*k2z+2*k3z+k4z);
}

int main() {
    srand(time(NULL));
    int sz[] = {3, 32, 32, 3};
    DNN *dnn = init_dnn(4, sz);

    long double xc=1, yc=1, zc=1, xq=1, yq=1, zq=1, dt=0.01L;

    printf("--- Deep Learning (DNN) Lorenz Quantization ---\n");
    for (int i = 0; i < EPOCHS; i++) {
        long double in[3] = {xc/30.0L, yc/30.0L, zc/30.0L};
        long double tar[3] = {xq/30.0L, yq/30.0L, zq/30.0L};
        forward(dnn, in);
        train(dnn, tar);
        rk4(lorenz_c, &xc, &yc, &zc, dt);
        rk4(lorenz_q, &xq, &yq, &zq, dt);
        if(i%2000==0) printf("Epoch %d: Training in progress...\n", i);
    }

    printf("\nTest Results (First 10 steps):\nTime, Classical_X, Quantum_X, DNN_Predicted_X, Error\n");
    xc=1; yc=1; zc=1; xq=1; yq=1; zq=1;
    for (int i = 0; i < 10; i++) {
        long double in[3] = {xc/30.0L, yc/30.0L, zc/30.0L};
        forward(dnn, in);
        long double pred_x = dnn->n[3][0] * 30.0L;
        printf("%.2Lf, %.6Lf, %.6Lf, %.6Lf, %.6Lf\n", i*dt, xc, xq, pred_x, fabsl(xq-pred_x));
        rk4(lorenz_c, &xc, &yc, &zc, dt);
        rk4(lorenz_q, &xq, &yq, &zq, dt);
    }
    return 0;
}
