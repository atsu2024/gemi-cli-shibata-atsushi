#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Simple Neural Network using long double for high precision
// Structure: 2 Inputs -> 2 Hidden Neurons -> 1 Output

typedef struct {
    long double input[2];
    long double hidden_weights[2][2];
    long double hidden_bias[2];
    long double hidden_output[2];
    long double output_weights[2];
    long double output_bias;
    long double output;
} MLP;

long double sigmoid_ld(long double x) {
    return 1.0L / (1.0L + expl(-x));
}

long double sigmoid_derivative_ld(long double x) {
    return x * (1.0L - x);
}

void init_mlp(MLP *net) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            net->hidden_weights[i][j] = ((long double)rand() / (long double)RAND_MAX) * 2.0L - 1.0L;
        }
        net->hidden_bias[i] = ((long double)rand() / (long double)RAND_MAX) * 2.0L - 1.0L;
        net->output_weights[i] = ((long double)rand() / (long double)RAND_MAX) * 2.0L - 1.0L;
    }
    net->output_bias = ((long double)rand() / (long double)RAND_MAX) * 2.0L - 1.0L;
}

void forward(MLP *net, long double in1, long double in2) {
    net->input[0] = in1;
    net->input[1] = in2;

    for (int i = 0; i < 2; i++) {
        long double activation = net->hidden_bias[i];
        for (int j = 0; j < 2; j++) {
            activation += net->input[j] * net->hidden_weights[j][i];
        }
        net->hidden_output[i] = sigmoid_ld(activation);
    }

    long double out_activation = net->output_bias;
    for (int i = 0; i < 2; i++) {
        out_activation += net->hidden_output[i] * net->output_weights[i];
    }
    net->output = sigmoid_ld(out_activation);
}

void train(MLP *net, long double target, long double lr) {
    // Backpropagation
    long double output_error = target - net->output;
    long double output_delta = output_error * sigmoid_derivative_ld(net->output);

    long double hidden_deltas[2];
    for (int i = 0; i < 2; i++) {
        long double hidden_error = output_delta * net->output_weights[i];
        hidden_deltas[i] = hidden_error * sigmoid_derivative_ld(net->hidden_output[i]);
    }

    // Update weights and biases
    for (int i = 0; i < 2; i++) {
        net->output_weights[i] += lr * output_delta * net->hidden_output[i];
        for (int j = 0; j < 2; j++) {
            net->hidden_weights[j][i] += lr * hidden_deltas[i] * net->input[j];
        }
        net->hidden_bias[i] += lr * hidden_deltas[i];
    }
    net->output_bias += lr * output_delta;
}

int main() {
    srand(time(NULL));
    MLP net;
    init_mlp(&net);

    // XOR dataset
    long double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    long double targets[4] = {0, 1, 1, 0};

    printf("Starting training (long double precision)...\n");
    for (int epoch = 0; epoch < 100000; epoch++) {
        for (int i = 0; i < 4; i++) {
            forward(&net, inputs[i][0], inputs[i][1]);
            train(&net, targets[i], 0.1L);
        }
        if (epoch % 10000 == 0) {
            long double total_error = 0;
            for (int i = 0; i < 4; i++) {
                forward(&net, inputs[i][0], inputs[i][1]);
                total_error += fabsl(targets[i] - net.output);
            }
            printf("Epoch %d, Avg Error: %0.15Lf\n", epoch, total_error / 4.0L);
        }
    }

    printf("\nFinal Results:\n");
    for (int i = 0; i < 4; i++) {
        forward(&net, inputs[i][0], inputs[i][1]);
        printf("In: %d,%d Target: %d Predict: %0.15Lf\n", 
               (int)inputs[i][0], (int)inputs[i][1], (int)targets[i], net.output);
    }

    return 0;
}
