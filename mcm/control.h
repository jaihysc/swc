#ifndef CONTROL_H
#define CONTROL_H

// Master control FSM
// Each control FSM state has a corresponding status LED pattern

typedef enum
{
    // Enter states wait for hardware in previous state to be disabled

    // Startup calibration
    CONTROL_CAL_ENTER = 0,
    CONTROL_CAL_SOSC_0,
    CONTROL_CAL_SOSC_1,
    CONTROL_CAL_MOT_0,
    CONTROL_CAL_MOT_1,

    // Runtime
    CONTROL_IDLE,
    CONTROL_SWEEP_THETA,
    CONTROL_SWEEP_RADIUS,
    CONTROL_CHARGING,
} ControlState;

#endif // CONTROL_H