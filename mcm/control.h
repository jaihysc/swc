#ifndef CONTROL_H
#define CONTROL_H

// Master control FSM
// Each control FSM state has a corresponding status LED pattern

typedef enum
{
    // Enter states wait for hardware in previous state to be disabled

    // Startup calibration
    CONTROL_CAL_ENTER = 0,
    CONTROL_CAL_0,
    CONTROL_CAL_1,

    // Runtime
    CONTROL_IDLE,
    CONTROL_SWEEP_THETA_ENTER,
    CONTROL_SWEEP_THETA,
    CONTROL_SWEEP_RADIUS_ENTER,
    CONTROL_SWEEP_RADIUS,
    CONTROL_CHARGING,
    CONTROL_RESET_RADIUS_ENTER,
    CONTROL_RESET_RADIUS,
    CONTROL_RESET_THETA_ENTER,
    CONTROL_RESET_THETA,

    CONTROL_TEST_MOTOR_0,
    CONTROL_TEST_MOTOR_1,
} ControlState;

#endif // CONTROL_H