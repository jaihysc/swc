#ifndef CONTROL_H
#define CONTROL_H

// Master control FSM
// Each control FSM state has a corresponding status LED pattern

typedef enum
{
    // Startup calibration
    CONTROL_CAL_ENTER = 0,
    CONTROL_CAL_0,
    CONTROL_CAL_1_ENTER,
    CONTROL_CAL_1,

    // Runtime
    CONTROL_IDLE_ENTER, // Enter states wait for hardware in previous state to be disabled
    CONTROL_IDLE,
    CONTROL_SWEEP_THETA_ENTER,
    CONTROL_SWEEP_THETA,
    CONTROL_SWEEP_RADIUS_ENTER,
    CONTROL_SWEEP_RADIUS,
} ControlState;

#endif // CONTROL_H