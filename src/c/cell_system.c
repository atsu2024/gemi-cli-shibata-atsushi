#include <stdio.h>
#include <math.h>
#include <string.h>

typedef long double precision_t;

typedef struct {
    precision_t Kp, Ki, Kd;
    precision_t integral;
    precision_t prev_error;
} BusinessPID;

typedef struct {
    char name[32];
    precision_t count;
    precision_t processing_power;
    precision_t jpy_cost;
} CellEntity;

precision_t run_pid(BusinessPID *pid, precision_t target, precision_t current) {
    precision_t error = target - current;
    pid->integral += error;
    precision_t derivative = error - pid->prev_error;
    pid->prev_error = error;
    return (pid->Kp * error) + (pid->Ki * pid->integral) + (pid->Kd * derivative);
}

int main() {
    precision_t cell_junior_count = 1.0e50L; 
    
    CellEntity perfect_cell = {"Perfect Cell", 1.0e12L, 1.0e20L, 5000.0L};
    CellEntity cell_junior = {"Cell Junior", cell_junior_count, 1.0e10L, 0.1L};

    BusinessPID biz_pid = {0.8L, 0.2L, 0.1L, 0.0L, 0.0L};
    
    precision_t target_break_even = 0.0L;
    precision_t total_jpy_balance = 0.0L;
    precision_t total_tasks_completed = 0.0L;

    printf("=== Universal Business Simulation (Atsushi Shibata Model) ===\n");
    printf("Targeting: %.0Lf Cell Juniors\n\n", (double)cell_junior.count);
    printf("%-6s | %-15s | %-15s | %-10s\n", "100m", "Balance (JPY)", "Total Tasks", "PID Out");
    printf("------------------------------------------------------------\n");

    for (int step = 1; step <= 20; step++) {
        precision_t step_tasks = (perfect_cell.count * perfect_cell.processing_power) + 
                                 (cell_junior.count * cell_junior.processing_power);
        total_tasks_completed += step_tasks;

        precision_t revenue = step_tasks * 1.0e-25L;
        precision_t expenses = (perfect_cell.count * perfect_cell.jpy_cost) + 
                               (cell_junior.count * cell_junior.jpy_cost);
        
        total_jpy_balance += (revenue - expenses);
        precision_t pid_output = run_pid(&biz_pid, target_break_even, total_jpy_balance);

        printf("%4dm  | %15.2Le | %15.2Le | %10.4Lf\n", 
               step * 100, (double)total_jpy_balance, (double)total_tasks_completed, (double)pid_output);

        if (pid_output > 0) cell_junior.processing_power *= (1.0L + pid_output * 1.0e-5L);
        cell_junior.count *= 1.05L;
    }

    printf("------------------------------------------------------------\n");
    printf("Final Business Analysis:\n");
    printf("Completed Tasks: %.4Le operations\n", (double)total_tasks_completed);
    printf("Final Balance  : %.4Le JPY\n", (double)total_jpy_balance);
    printf("System Status  : Optimal (Business Calculations Finished)\n");

    return 0;
}
