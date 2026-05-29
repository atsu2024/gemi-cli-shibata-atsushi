#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define VECTOR_SIZE 10
#define MAX_ENTRIES 50
#define MAX_CONTENT 256

typedef struct {
    int id;
    char content[MAX_CONTENT];
    long double vector[VECTOR_SIZE];
} KnowledgeBaseEntry;

// Calculate Cosine Similarity using long double for high precision
long double cosine_similarity(long double *v1, long double *v2, int size) {
    long double dot_product = 0.0L;
    long double norm_v1 = 0.0L;
    long double norm_v2 = 0.0L;

    for (int i = 0; i < size; i++) {
        dot_product += v1[i] * v2[i];
        norm_v1 += v1[i] * v1[i];
        norm_v2 += v2[i] * v2[i];
    }

    if (norm_v1 <= 0.0L || norm_v2 <= 0.0L) return 0.0L;
    return dot_product / (sqrtl(norm_v1) * sqrtl(norm_v2));
}

// Simple JSON-like parser for the specific format provided
int load_data(const char *filename, KnowledgeBaseEntry *db) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;

    char line[1024];
    int count = 0;
    while (fgets(line, sizeof(line), fp) && count < MAX_ENTRIES) {
        char *id_pos = strstr(line, "\"id\":");
        if (id_pos) {
            sscanf(id_pos, "\"id\": %d,", &db[count].id);
            
            // Read content
            if (fgets(line, sizeof(line), fp)) {
                char *start = strchr(line, ':');
                if (start) {
                    start = strchr(start, '"');
                    if (start) {
                        start++; // Skip '"'
                        char *end = strrchr(line, '"');
                        if (end) {
                            *end = '\0';
                            strncpy(db[count].content, start, MAX_CONTENT - 1);
                            db[count].content[MAX_CONTENT - 1] = '\0';
                        }
                    }
                }
            }
            
            // Read vector
            if (fgets(line, sizeof(line), fp)) {
                char *vstart = strchr(line, '[');
                if (vstart) {
                    vstart++;
                    char *ptr;
                    for (int j = 0; j < VECTOR_SIZE; j++) {
                        db[count].vector[j] = strtold(vstart, &ptr);
                        vstart = ptr;
                        if (*vstart == ',' || *vstart == ']') vstart++;
                    }
                }
            }
            count++;
        }
    }
    fclose(fp);
    return count;
}

int main(int argc, char *argv[]) {
    KnowledgeBaseEntry db[MAX_ENTRIES];
    int count = load_data("rag_research.json", db);

    if (count == 0) {
        fprintf(stderr, "Error: Could not load rag_research.json or file empty.\n");
        return 1;
    }

    long double query_vector[VECTOR_SIZE];
    if (argc >= VECTOR_SIZE + 1) {
        for (int i = 0; i < VECTOR_SIZE; i++) {
            query_vector[i] = strtold(argv[i + 1], NULL);
        }
    } else {
        fprintf(stderr, "No query vector provided. Using default [0.5, ...].\n");
        for (int i = 0; i < VECTOR_SIZE; i++) {
            query_vector[i] = 0.5L;
        }
    }

    int best_match = -1;
    long double max_sim = -2.0L;

    for (int i = 0; i < count; i++) {
        long double sim = cosine_similarity(query_vector, db[i].vector, VECTOR_SIZE);
        if (sim > max_sim) {
            max_sim = sim;
            best_match = i;
        }
    }

    if (best_match != -1) {
        // Output ONLY the ID to stdout for the server to read
        printf("%d\n", db[best_match].id);
        
        // Log details to stderr
        fprintf(stderr, "Match ID: %d, Sim: %.20Lf\n", db[best_match].id, max_sim);
    }

    return 0;
}
