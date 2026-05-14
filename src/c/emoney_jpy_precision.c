#include <stdio.h>
#include <string.h>
#include <time.h>

#define MAX_HISTORY 10
#define REWARD_RATE 0.01L // 1% points reward

typedef enum {
    TRANSACTION_CHARGE,
    TRANSACTION_PAYMENT
} TransactionType;

typedef struct {
    TransactionType type;
    long double amount;
    char timestamp[20];
} History;

typedef struct {
    char user_id[32];
    long double balance;
    long double points;
    History history[MAX_HISTORY];
    int history_count;
} JPYElectronicMoney;

void get_now_timestamp(char *buf) {
    time_t t = time(NULL);
    struct tm *tm_info = localtime(&t);
    if (tm_info) {
        strftime(buf, 20, "%Y/%m/%d %H:%M:%S", tm_info);
    } else {
        strcpy(buf, "0000/00/00 00:00:00");
    }
}

void add_history(JPYElectronicMoney *sys, TransactionType type, long double amount) {
    if (sys->history_count >= MAX_HISTORY) {
        for (int i = 0; i < MAX_HISTORY - 1; i++) {
            sys->history[i] = sys->history[i + 1];
        }
        sys->history_count = MAX_HISTORY - 1;
    }
    
    sys->history[sys->history_count].type = type;
    sys->history[sys->history_count].amount = amount;
    get_now_timestamp(sys->history[sys->history_count].timestamp);
    sys->history_count++;
}

void init_system(JPYElectronicMoney *sys, const char *id) {
    sys->balance = 0.0L;
    sys->points = 0.0L;
    sys->history_count = 0;
    memset(sys->user_id, 0, sizeof(sys->user_id));
    strncpy(sys->user_id, id, sizeof(sys->user_id) - 1);
}

void charge(JPYElectronicMoney *sys, long double amount) {
    if (amount <= 0) return;
    sys->balance += amount;
    add_history(sys, TRANSACTION_CHARGE, amount);
    printf("[CHARGE] %.0f JPY credited.\n", (double)amount);
}

int pay(JPYElectronicMoney *sys, long double amount) {
    if (amount <= 0) return 0;
    if (sys->balance < amount) {
        printf("[ERROR] Insufficient balance. Required: %.0f JPY / Balance: %.0f JPY\n", (double)amount, (double)sys->balance);
        return 0;
    }
    
    sys->balance -= amount;
    long double earned_points = amount * REWARD_RATE;
    sys->points += earned_points;
    
    add_history(sys, TRANSACTION_PAYMENT, amount);
    printf("[PAYMENT] %.0f JPY paid. (Points earned: %.2f)\n", (double)amount, (double)earned_points);
    return 1;
}

void show_status(JPYElectronicMoney *sys) {
    printf("\n--- Account Status [%s] ---\n", sys->user_id);
    printf("Balance: %15.0f JPY\n", (double)sys->balance);
    printf("Points:  %15.2f pt\n", (double)sys->points);
    printf("------------------------------\n");
    printf("Recent History:\n");
    for (int i = 0; i < sys->history_count; i++) {
        printf("  [%s] %s: %.0f JPY\n", 
            sys->history[i].timestamp, 
            sys->history[i].type == TRANSACTION_CHARGE ? "CHARGE " : "PAYMENT",
            (double)sys->history[i].amount);
    }
    printf("------------------------------\n\n");
}

int main() {
    JPYElectronicMoney wallet;
    init_system(&wallet, "G-CLI-USER-2026");

    printf("=== JPY Electronic Money System (High Precision) ===\n");

    charge(&wallet, 50000.0L);
    show_status(&wallet);

    pay(&wallet, 1280.0L);
    pay(&wallet, 3500.0L);
    show_status(&wallet);

    charge(&wallet, 10000.0L);
    pay(&wallet, 60000.0L); // Failure case
    
    show_status(&wallet);

    return 0;
}
