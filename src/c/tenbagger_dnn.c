#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT_SIZE 3   
#define HIDDEN_SIZE 5
#define OUTPUT_SIZE 1
#define NUM_STOCKS 6

typedef struct {
    char code[16];
    char name[64];
    long double features[INPUT_SIZE]; 
    long double score;
} Stock;

long double activation(long double x) {
    return 1.0L / (1.0L + expl(-x)); // Sigmoid for 0.0-1.0 range
}

long double predict_score(long double* input, long double weights1[INPUT_SIZE][HIDDEN_SIZE], long double weights2[HIDDEN_SIZE][OUTPUT_SIZE]) {
    long double hidden[HIDDEN_SIZE] = {0};
    long double output = 0;

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            hidden[j] += input[i] * weights1[i][j];
        }
        hidden[j] = activation(hidden[j]);
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        output += hidden[j] * weights2[j][0];
    }
    return activation(output);
}

int main() {
    Stock stocks[NUM_STOCKS] = {
        {"5595", "QPS Institute (JP)", {0.85L, 0.70L, 0.90L}, 0},
        {"9553", "MicroAd (JP)", {0.95L, 0.85L, 0.75L}, 0},
        {"9235", "UreruNet (JP)", {0.98L, 0.60L, 0.80L}, 0},
        {"SMCI", "Super Micro (US)", {0.40L, 0.50L, 0.95L}, 0},
        {"NVDA", "NVIDIA (US)", {0.10L, 0.30L, 0.99L}, 0},
        {"PLTR", "Palantir (US)", {0.55L, 0.65L, 0.88L}, 0}
    };

    long double w1[INPUT_SIZE][HIDDEN_SIZE] = {
        {0.2L, 0.1L, 0.3L, 0.1L, 0.2L},
        {0.1L, 0.4L, 0.2L, 0.3L, 0.1L},
        {0.4L, 0.2L, 0.5L, 0.1L, 0.3L}
    };
    long double w2[HIDDEN_SIZE][OUTPUT_SIZE] = {
        {0.3L}, {0.2L}, {0.4L}, {0.1L}, {0.2L}
    };

    printf("========================================================\n");
    printf("  DNN Tenbagger Prediction (long double Precision)\n");
    printf("  Target: 1376partners.com Attention Stocks\n");
    printf("========================================================\n");
    printf("%-8s %-20s %-15s\n", "Code", "Name", "Potential Score");
    printf("--------------------------------------------------------\n");

    for (int i = 0; i < NUM_STOCKS; i++) {
        stocks[i].score = predict_score(stocks[i].features, w1, w2);
        
        // スコアを 0-100% にスケーリング
        long double final_score = (stocks[i].score + 1.0L) * 50.0L;
        
        printf("%-8s %-20s %10.2Lf%%\n", stocks[i].code, stocks[i].name, final_score);
    }
    printf("========================================================\n");
    printf("Analysis Complete: Higher %% indicates stronger DNN signal.\n");

    return 0;
}
