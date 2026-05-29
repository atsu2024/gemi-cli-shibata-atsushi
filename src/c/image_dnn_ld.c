#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <errno.h>

#define INPUT_SIZE 1024
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 1024
#define LEARNING_RATE 0.01L
#define EPOCHS 1000

typedef struct {
    long double input[INPUT_SIZE];
    long double w1[INPUT_SIZE][HIDDEN_SIZE];
    long double b1[HIDDEN_SIZE];
    long double h_out[HIDDEN_SIZE];
    long double w2[HIDDEN_SIZE][OUTPUT_SIZE];
    long double b2[OUTPUT_SIZE];
    long double output[OUTPUT_SIZE];
} Autoencoder;

long double sigmoid_ld(long double x) {
    return 1.0L / (1.0L + expl(-x));
}

long double sigmoid_derivative_ld(long double x) {
    return x * (1.0L - x);
}

void init_network(Autoencoder *net) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->w1[i][j] = ((long double)rand() / RAND_MAX) * 0.2L - 0.1L;
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        net->b1[i] = 0.0L;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            net->w2[i][j] = ((long double)rand() / RAND_MAX) * 0.2L - 0.1L;
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        net->b2[i] = 0.0L;
    }
}

void forward(Autoencoder *net, long double *input) {
    // Input to Hidden
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        long double act = net->b1[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            act += input[i] * net->w1[i][j];
        }
        net->h_out[j] = sigmoid_ld(act);
    }
    // Hidden to Output
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        long double act = net->b2[j];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            act += net->h_out[i] * net->w2[i][j];
        }
        net->output[j] = sigmoid_ld(act);
    }
}

void train(Autoencoder *net, long double *input) {
    forward(net, input);

    // Output layer errors
    long double out_deltas[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        long double error = input[i] - net->output[i];
        out_deltas[i] = error * sigmoid_derivative_ld(net->output[i]);
    }

    // Hidden layer errors
    long double h_deltas[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        long double error = 0.0L;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            error += out_deltas[j] * net->w2[i][j];
        }
        h_deltas[i] = error * sigmoid_derivative_ld(net->h_out[i]);
    }

    // Update weights and biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            net->w2[i][j] += LEARNING_RATE * out_deltas[j] * net->h_out[i];
        }
    }
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        net->b2[j] += LEARNING_RATE * out_deltas[j];
    }
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->w1[i][j] += LEARNING_RATE * h_deltas[j] * input[i];
        }
    }
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        net->b1[j] += LEARNING_RATE * h_deltas[j];
    }
}

int main() {
    srand(time(NULL));
    static Autoencoder net; // Use static to avoid stack overflow for large struct
    long double input_data[INPUT_SIZE];

    // Load data from CSV
    FILE *fp = fopen("image_data.csv", "r");
    if (!fp) {
        printf("Error: Could not open image_data.csv\n");
        return 1;
    }
    for (int i = 0; i < INPUT_SIZE; i++) {
        char token[128];
        int len = 0;
        int ch;

        while ((ch = fgetc(fp)) != EOF && (ch == ',' || ch == '\n' || ch == '\r' || ch == '\t' || ch == ' ')) {
        }
        if (ch == EOF) break;

        do {
            if (len < (int)sizeof(token) - 1) token[len++] = (char)ch;
            ch = fgetc(fp);
        } while (ch != EOF && ch != ',' && ch != '\n' && ch != '\r' && ch != '\t' && ch != ' ');

        token[len] = '\0';
        errno = 0;
        char *endptr = NULL;
        input_data[i] = strtold(token, &endptr);
        if (errno != 0 || endptr == token || *endptr != '\0') {
            printf("Error: Invalid value in image_data.csv at index %d\n", i);
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);

    init_network(&net);

    printf("Starting DNN training with long double precision...\n");
    for (int epoch = 0; epoch <= EPOCHS; epoch++) {
        train(&net, input_data);
        if (epoch % 100 == 0) {
            long double mse = 0;
            for (int i = 0; i < INPUT_SIZE; i++) {
                mse += powl(input_data[i] - net.output[i], 2);
            }
            printf("Epoch %d, MSE: %0.15f\n", epoch, (double)(mse / INPUT_SIZE));
        }
    }

    // Save reconstructed output for verification
    fp = fopen("output_data.csv", "w");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        fprintf(fp, "%Lf\n", net.output[i]);
    }
    fclose(fp);
    printf("Reconstructed image data saved to output_data.csv\n");

    return 0;
}
