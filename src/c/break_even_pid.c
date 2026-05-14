#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    long double kp;
    long double ki;
    long double kd;
    long double integral;
    long double prev_error;
} PIDController;

long double calculate_pid(PIDController *pid, long double setpoint, long double pv, long double dt) {
    long double error = setpoint - pv;
    pid->integral += error * dt;
    long double derivative = (error - pid->prev_error) / dt;
    long double output = pid->kp * error + pid->ki * pid->integral + pid->kd * derivative;
    pid->prev_error = error;
    return output;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <target_output>\n", argv[0]);
        return 1;
    }

    char *endptr;
    long double target = strtold(argv[1], &endptr);
    if (argv[1] == endptr) {
        fprintf(stderr, "Invalid input: %s\n", argv[1]);
        return 1;
    }

    // System: y = 10*u + 1. System gain is 10.
    // To stabilize, Kp should be small. 
    PIDController pid = {0.01L, 0.005L, 0.001L, 0.000L, 0.000L}; 
    long double current_u = 0.0L;
    long double dt = 0.1L;

    printf("Simulation for Target: %Lf (100m units)\n", target);
    printf("Step, Control_Input(u), System_Output(y), Error\n");

    for (int i = 0; i < 50; i++) {
        long double y = current_u * 10.0L + 1.0L;
        long double error = target - y;

        printf("%d, %15.6Lf, %15.6Lf, %15.6Lf\n", i, current_u, y, error);

        if (fabsl(error) < 1e-9L) break;

        long double delta_u = calculate_pid(&pid, target, y, dt);
        current_u += delta_u;
    }

    return 0;
}
