#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/**
 * dnn_all_data_long_double.c
 * 
 * 物理法則（ビオ・サバール）と経済ロジック（ポイント還元）を
 * 深層学習（DNN）に統合する高精度 C 言語プログラム。
 * 全てのデータを long double 型で処理し、DNN（多層ニューラルネットワーク）によりモデル化します。
 */

typedef struct {
    int num_layers;
    int *layer_sizes;
    long double **nodes;
    long double ***weights;
    long double **biases;
    long double **deltas;
} DNN;

// 活性化関数: 高精度 Sigmoid
long double sigmoid_ld(long double x) {
    return 1.0L / (1.0L + expl(-x));
}

// 活性化関数の微分
long double sigmoid_derivative_ld(long double x) {
    return x * (1.0L - x);
}

// DNNの初期化
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
            dnn->biases[i] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
            dnn->weights[i-1] = (long double**)malloc(layer_sizes[i-1] * sizeof(long double*));
            for (int j = 0; j < layer_sizes[i-1]; j++) {
                dnn->weights[i-1][j] = (long double*)malloc(layer_sizes[i] * sizeof(long double));
                for (int k = 0; k < layer_sizes[i]; k++) {
                    // 重みの初期化 (Xavier的な初期化の簡易版)
                    dnn->weights[i-1][j][k] = (((long double)rand() / RAND_MAX) * 2.0L - 1.0L) * sqrtl(2.0L / layer_sizes[i-1]);
                }
            }
            for (int j = 0; j < layer_sizes[i]; j++) {
                dnn->biases[i][j] = 0.0L;
            }
        }
    }
    return dnn;
}

// 順伝播
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

// 逆伝播
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

// --- データ生成エンジン (既存の全ロジックをDNN用へ) ---

// 1. 物理データ: ビオ・サバールの法則の近似学習用
// 入力: [距離r, 電流I], 出力: [磁場H]
void generate_physics_data(long double *input, long double *target) {
    long double r = input[0] * 0.1L + 0.01L; // 0.01m ~ 0.11m
    long double I = input[1] * 10.0L;        // 0A ~ 10A
    const long double PI_L = acosl(-1.0L);
    
    // H = I / (2 * PI * r)
    long double H = I / (2.0L * PI_L * r);
    
    // ターゲットの正規化 (0.0 ~ 1.0 に収まるように Sigmoid 的に圧縮)
    target[0] = sigmoid_ld(H / 50.0L); 
}

// 2. 経済データ: ポイント還元の学習用
// 入力: [金額, 利用回数], 出力: [獲得ポイント]
void generate_finance_data(long double *input, long double *target) {
    long double amount = input[0] * 10000.0L;
    long double count  = input[1] * 10.0L;
    long double reward_rate = 0.01L;
    
    long double points = amount * reward_rate * (1.0L + count * 0.05L);
    
    target[0] = sigmoid_ld(points / 100.0L);
}

int main() {
    srand((unsigned int)time(NULL));
    printf("====================================================\n");
    printf("   Unified Deep Neural Network (DNN) long double    \n");
    printf("====================================================\n");

    // ネットワーク構成: 入力2 -> 隠れ16 -> 隠れ16 -> 出力1
    int layers_config[] = {2, 16, 16, 1};
    int num_layers = sizeof(layers_config) / sizeof(layers_config[0]);
    DNN *dnn = create_dnn(num_layers, layers_config);

    int num_samples = 1000;
    int epochs = 20000;
    long double learning_rate = 0.15L;

    printf("Network Structure: ");
    for(int i=0; i<num_layers; i++) printf("%d ", layers_config[i]);
    printf("\nTraining data: Physics + Finance integrated.\n");
    printf("Starting training for %d epochs...\n\n", epochs);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        long double mse = 0;
        for (int i = 0; i < num_samples; i++) {
            long double in[2] = {(long double)rand()/RAND_MAX, (long double)rand()/RAND_MAX};
            long double tar[1];
            
            // 物理と経済のデータを交互に学習
            if (i % 2 == 0) generate_physics_data(in, tar);
            else           generate_finance_data(in, tar);

            forward_prop(dnn, in);
            back_prop(dnn, tar, learning_rate);
            
            long double err = tar[0] - dnn->nodes[num_layers-1][0];
            mse += err * err;
        }

        if (epoch % 2000 == 0 || epoch == 1) {
            printf("Epoch [%5d/%5d] MSE: %.15Lf\n", epoch, epochs, mse / num_samples);
        }
    }

    printf("\n--- Final Validation (Deep Learning Test) ---\n");
    // テスト1: 物理
    long double p_in[2] = {0.05L, 5.0L}; // r=0.06m, I=5A
    long double p_tar[1];
    generate_physics_data(p_in, p_tar);
    forward_prop(dnn, p_in);
    printf("Physics (Biot-Savart) -> Target: %.8Lf, DNN Prediction: %.8Lf\n", p_tar[0], dnn->nodes[num_layers-1][0]);

    // テスト2: 経済
    long double f_in[2] = {0.8L, 0.2L}; // amount=8000, count=2
    long double f_tar[1];
    generate_finance_data(f_in, f_tar);
    forward_prop(dnn, f_in);
    printf("Finance (Reward Pt)   -> Target: %.8Lf, DNN Prediction: %.8Lf\n", f_tar[0], dnn->nodes[num_layers-1][0]);

    printf("\nConversion to Deep Learning completed successfully.\n");

    return 0;
}
