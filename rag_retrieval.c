#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int id;
    const char* content;
    long double vector[3];
} KnowledgeBase;

KnowledgeBase kb[] = {
    {1, "The Biot-Savart law relates magnetic fields to the currents which are their sources.", {0.1234567890123456789L, 0.9876543210987654321L, 0.5555555555555555555L}},
    {2, "The Lorenz system is a system of ordinary differential equations first studied by Edward Lorenz.", {0.9999999999999999999L, 0.1111111111111111111L, 0.2222222222222222222L}},
    {3, "Deep Neural Networks use multiple layers to progressively extract higher-level features from raw input.", {0.4444444444444444444L, 0.4444444444444444444L, 0.8888888888888888888L}}
};

long double calculate_distance(long double v1[3], long double v2[3]) {
    long double dist = 0;
    for (int i = 0; i < 3; i++) {
        long double diff = v1[i] - v2[i];
        dist += diff * diff;
    }
    return sqrtl(dist);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <v1> <v2> <v3>\n", argv[0]);
        return 1;
    }

    long double query[3];
    query[0] = strtold(argv[1], NULL);
    query[1] = strtold(argv[2], NULL);
    query[2] = strtold(argv[3], NULL);

    int best_id = -1;
    long double min_dist = -1;

    for (int i = 0; i < 3; i++) {
        long double dist = calculate_distance(query, kb[i].vector);
        if (min_dist < 0 || dist < min_dist) {
            min_dist = dist;
            best_id = kb[i].id;
        }
    }

    printf("%d\n", best_id);
    return 0;
}
