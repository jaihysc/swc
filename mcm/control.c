#include <hardware/gpio.h>

#include "control.h"
#include "dac.h"
#include "hw_config.h"
#include "motor.h"
#include "sosc.h"
#include "status.h"

enum
{
    THETA_MOT                 = 1, // Motor/DAC for theta sweep, must be 1 since only DAC1 can run with SOSC1 when sweeping
    RADIUS_MOT                = 0,
    // THETA_SWEEP_STEPS      = 6000, // Steps for full rotation
    // RADIUS_SWEEP_STEPS     = 1200  // steps for full radius
    THETA_SWEEP_STEP_MAX      = 250,
    THETA_SWEEP_STEP          = 24,
    RADIUS_SWEEP_STEP_MAX     = 60,
    RADIUS_SWEEP_STEP         = 20,

    CHARGE_DET_VOTE           = 1 << 17, // FSM iterations to declare charging
    CHARGE_DET_TIMEOUT        = 1 << 18, // FSM iterations to check for charging
};

typedef enum
{
    // Enter states wait for hardware in previous state to be disabled
    // Four main states
    // Cal   -> Idle
    //  ^        v
    // Reset <- Charging

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

    // CONTROL_RESET_MOT_0,
    // CONTROL_RESET_MOT_1,
} ControlState;

typedef struct
{
    // Sweep steps / current direction
    int32_t countDiffPrev;
    int32_t countDiffAccum;
    uint32_t chargeDetTimeout;
    uint32_t chargeVote;
    ControlState state;

    int16_t thetaSteps;
    int16_t radiusSteps;
    uint8_t measIter;
    uint8_t nearPhone : 1; // Theta/radius sweep near phone
    uint8_t enterFine : 1; // Radius sweep switches from coarse to fine mode
    uint8_t sweepDir  : 1;
    uint8_t reserved  : 5;
} Control;

Control control;

// Tries to move theta, true if moved
static bool stepTheta(int16_t stepMul) {
    int16_t newStep = control.thetaSteps + stepMul;

    // Clamp
    if (newStep < 0) {
        stepMul = -1 * control.thetaSteps;
    }
    else if (newStep > THETA_SWEEP_STEP_MAX) {
        stepMul = THETA_SWEEP_STEP_MAX - control.thetaSteps;
    }

    if (stepMul == 0) {
        return false; // Can't move
    }
    else {
        motorMove(THETA_MOT, THETA_SWEEP_STEP * stepMul);
        control.thetaSteps += stepMul;
        return true;
    }
}

static bool stepRadius(int16_t stepMul) {
    int16_t newStep = control.radiusSteps + stepMul;

    // Clamp
    if (newStep < 0) {
        stepMul = -1 * control.radiusSteps;
    }
    else if (newStep > RADIUS_SWEEP_STEP_MAX) {
        stepMul = RADIUS_SWEEP_STEP_MAX - control.radiusSteps;
    }

    if (stepMul == 0) {
        return false; // Can't move
    }
    else {
        motorMove(RADIUS_MOT, RADIUS_SWEEP_STEP * stepMul);
        control.radiusSteps += stepMul;
        return true;
    }
}

void runControl() {
    Control* fsm = &control;
    switch (fsm->state) {
        // Calibrate
        case CONTROL_CAL_ENTER:
        {
            if (!dacActive(0) && !dacActive(1)) {
                fsm->state = CONTROL_CAL_MOT_0;
            }
            break;
        }

        case CONTROL_CAL_MOT_0:
        {
            if (motorCalibrate(0)) {
                fsm->radiusSteps = 0;
                fsm->state = CONTROL_CAL_MOT_1;
            }
            break;
        }

        case CONTROL_CAL_MOT_1:
        {
            if (motorCalibrate(1)) {
                fsm->thetaSteps = 0;

                statusSet(STATUS_CAL_1);
                fsm->state = CONTROL_CAL_SOSC_1;
            }
            break;
        }

        case CONTROL_CAL_SOSC_1: // First calibrate SOSC1, then SOSC0, so SOSC0 can remain on for idle
        {
            if (soscCalibrate(1)) {
                statusSet(STATUS_CAL_1);
                fsm->state = CONTROL_CAL_SOSC_0;
            }
            break;
        }

        case CONTROL_CAL_SOSC_0:
        {
            if (soscCalibrate(0)) {
                statusSet(STATUS_IDLE);
                fsm->state = CONTROL_IDLE;
            }
            break;
        }

        // Idle
        case CONTROL_IDLE:
        {
            bool detected;
            if (soscDetect(&detected, 0)) {
                if (detected) {
                    statusSet(STATUS_SWEEP_THETA);

                    fsm->state = CONTROL_SWEEP_THETA;
                }
            }
            break;
        }

        // Search for phone
        case CONTROL_SWEEP_THETA:
        {
            if (motorReady(THETA_MOT)) {
                int32_t countDiff;
                if (soscDelta(&countDiff, 1)) { // Only DAC1 and SOSC1 can be used at the same time
                    bool avgReady = false;
                    fsm->countDiffAccum += countDiff;
                    ++fsm->measIter;
                    if (fsm->measIter >= 2) {
                        countDiff = fsm->countDiffAccum / 2;

                        fsm->countDiffAccum = 0;
                        fsm->measIter = 0;
                        avgReady = true;
                    }

                    // Update based on measurement
                    if (avgReady) {
                        if (fsm->nearPhone && countDiff < fsm->countDiffPrev) {
                            // Peak found!
                            fsm->countDiffPrev = 0;
                            fsm->nearPhone = false;

                            // Take a few steps backwards since the charging coil and the sosc coil are offset
                            stepTheta(-10);

                            fsm->state = CONTROL_SWEEP_RADIUS;
                        }
                        else {
                            // When near the phone, step slower
                            if (countDiff > 30) {
                                fsm->nearPhone = true;
                            }

                            uint32_t step = (fsm->nearPhone) ? 1 : 10;
                            if (!stepTheta(step)) {
                                // Reached end but didn't find phone, give up
                                fsm->countDiffPrev = 0;
                                fsm->nearPhone = false;

                                fsm->state = CONTROL_CAL_ENTER;
                            }
                        }

                        fsm->countDiffPrev = countDiff;
                    }
                }
            }
            break;
        }

        case CONTROL_SWEEP_RADIUS:
        {
            if (motorReady(THETA_MOT) && motorReady(RADIUS_MOT)) {
                int32_t step = (fsm->nearPhone) ? 1 : 4;
                int32_t stepSign = (fsm->sweepDir == 0) ? 1 : -1;

                if (gpio_get(GPIO_CHARGING)) {
                    // When charging light first turns on, go back few steps and slowly step forwards
                    if (!fsm->nearPhone) {
                        stepRadius(-1 * stepSign * 4);
                        fsm->nearPhone = true;
                        fsm->enterFine = true; // Wait for timeout first before counting charge votes
                    }
                    else if (fsm->nearPhone && !fsm->enterFine) {
                        ++fsm->chargeVote;
                        if (fsm->chargeVote > CHARGE_DET_VOTE) { // Charging light needs to remain on, not flashing
                            fsm->sweepDir = 0;
                            fsm->chargeDetTimeout = 0;
                            fsm->chargeVote = 0;
                            fsm->nearPhone = false;

                            motorDisable(); // Turn off motors to save power

                            statusSet(STATUS_CHARGING);
                            fsm->state = CONTROL_CHARGING;
                        }
                    }
                }
                else {
                    fsm->chargeVote = 0;
                }

                // When first entering fine mode, need to wait extra time for charging light to clear
                uint32_t timeout = CHARGE_DET_TIMEOUT;
                if (fsm->enterFine) {
                    timeout *= 4;
                }

                // Didn't find phone, keep sweeping
                ++fsm->chargeDetTimeout;
                if (fsm->chargeDetTimeout > timeout) {
                    fsm->chargeDetTimeout = 0;
                    fsm->enterFine = false;

                    if (!stepRadius(step * stepSign)) {
                        // Didn't find the phone, move theta and try again
                        stepTheta(5);
                        fsm->sweepDir = ~fsm->sweepDir;
                        fsm->nearPhone = false;
                    }
                }
            }
            break;
        }

        // Charging
        case CONTROL_CHARGING:
        {
            if (!gpio_get(GPIO_CHARGING)) {
                ++fsm->chargeVote;
                if (fsm->chargeVote > CHARGE_DET_VOTE) {
                    fsm->chargeVote = 0;
                    fsm->state = CONTROL_CAL_ENTER;
                }
            }
            else {
                fsm->chargeVote = 0;
            }
            break;
        }

        // case CONTROL_RESET_MOT_0:
        // {
        //     if (motorCalibrate(0)) {
        //         fsm->radiusSteps = 0;
        //         fsm->state = CONTROL_RESET_MOT_1;
        //     }
        //     break;
        // }

        // case CONTROL_RESET_MOT_1:
        // {
        //     if (motorCalibrate(1)) {
        //         fsm->thetaSteps = 0;

        //         statusSet(STATUS_IDLE);
        //         fsm->state = CONTROL_IDLE;
        //     }
        //     break;
        // }

        default:
        {
            fsm->state = CONTROL_CAL_ENTER;
            break;
        }
    }
}
