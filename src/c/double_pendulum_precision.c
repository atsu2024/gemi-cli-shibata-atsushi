#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * Double Pendulum Simulation
 * Using Runge-Kutta 4th Order (RK4) Integration
 * High-Precision using long double
 */

typedef struct {
    long double g;      // gravity
    long double m1;     // mass 1
    long double m2;     // mass 2
    long double l1;     // length 1
    long double l2;     // length 2
} Params;

typedef struct {
    long double th1;    // angle 1
    long double th2;    // angle 2
    long double w1;     // angular velocity 1
    long double w2;     // angular velocity 2
} State;

void derivatives(State s, Params p, State *ds) {
    long double delta = s.th2 - s.th1;
    long double den1 = (2 * p.m1 + p.m2 - p.m2 * cosl(2 * delta));
    
    // dth1/dt
    ds->th1 = s.w1;
    // dth2/dt
    ds->th2 = s.w2;
    
    // dw1/dt
    long double num1 = -p.g * (2 * p.m1 + p.m2) * sinl(s.th1);
    long double num2 = -p.m2 * p.g * sinl(s.th1 - 2 * s.th2);
    long double num3 = -2 * sinl(delta) * p.m2 * (s.w2 * s.w2 * p.l2 + s.w1 * s.w1 * p.l1 * cosl(delta));
    ds->w1 = (num1 + num2 + num3) / (p.l1 * den1);
    
    // dw2/dt
    long double num4 = 2 * sinl(delta) * (s.w1 * s.w1 * p.l1 * (p.m1 + p.m2) + p.g * (p.m1 + p.m2) * cosl(s.th1) + s.w2 * s.w2 * p.l2 * p.m2 * cosl(delta));
    ds->w2 = num4 / (p.l2 * den1);
}

State rk4_step(State s, Params p, long double dt) {
    State k1, k2, k3, k4, res;
    State temp;
    
    derivatives(s, p, &k1);
    
    temp.th1 = s.th1 + k1.th1 * dt / 2.0L;
    temp.th2 = s.th2 + k1.th2 * dt / 2.0L;
    temp.w1  = s.w1  + k1.w1  * dt / 2.0L;
    temp.w2  = s.w2  + k1.w2  * dt / 2.0L;
    derivatives(temp, p, &k2);
    
    temp.th1 = s.th1 + k2.th1 * dt / 2.0L;
    temp.th2 = s.th2 + k2.th2 * dt / 2.0L;
    temp.w1  = s.w1  + k2.w1  * dt / 2.0L;
    temp.w2  = s.w2  + k2.w2  * dt / 2.0L;
    derivatives(temp, p, &k3);
    
    temp.th1 = s.th1 + k3.th1 * dt;
    temp.th2 = s.th2 + k3.th2 * dt;
    temp.w1  = s.w1  + k3.w1  * dt;
    temp.w2  = s.w2  + k3.w2  * dt;
    derivatives(temp, p, &k4);
    
    res.th1 = s.th1 + (dt / 6.0L) * (k1.th1 + 2 * k2.th1 + 2 * k3.th1 + k4.th1);
    res.th2 = s.th2 + (dt / 6.0L) * (k1.th2 + 2 * k2.th2 + 2 * k3.th2 + k4.th2);
    res.w1  = s.w1  + (dt / 6.0L) * (k1.w1  + 2 * k2.w1  + 2 * k3.w1  + k4.w1 );
    res.w2  = s.w2  + (dt / 6.0L) * (k1.w2  + 2 * k2.w2  + 2 * k3.w2  + k4.w2 );
    
    return res;
}

int main() {
    printf("----------------------------------------------------------------\n");
    printf("  Double Pendulum Simulation - High-Precision RK4 (long double)\n");
    printf("----------------------------------------------------------------\n");

    Params p = {9.81L, 1.0L, 1.0L, 1.0L, 1.0L};
    State s = {M_PI / 2.0L, M_PI / 2.0L, 0.0L, 0.0L}; // Initial: horizontal
    
    long double dt = 0.01L;
    long double t = 0.0L;
    int steps = 1000;

    printf("Time, Theta1, Theta2, Omega1, Omega2\n");
    for (int i = 0; i <= steps; i++) {
        if (i % 100 == 0) {
            printf("%.2Lf, %.10Lf, %.10Lf, %.10Lf, %.10Lf\n", t, s.th1, s.th2, s.w1, s.w2);
        }
        s = rk4_step(s, p, dt);
        t += dt;
    }

    printf("\n[Success] Simulation completed.\n");
    return 0;
}
