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

    THETA_SWEEP_STEP_MAX      = 260, // Number of stepper motor steps, for each "theta step"
    THETA_SWEEP_STEP          = 24,
    RADIUS_SWEEP_STEP_MAX     = 60,
    RADIUS_SWEEP_STEP         = 20,

    THETA_COARSE_STEP         = 20, // Coarse steps are 10x the minimum step size
    THETA_COARSE_STEP_MAX     = THETA_SWEEP_STEP_MAX / THETA_COARSE_STEP,
    THETA_ADJUST_STEP         = 5, // When radius sweep couldn't find phone

    THETA_ADJUST_MAX          = 16,

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
    CONTROL_CAL_MOT_0,
    CONTROL_CAL_MOT_1,

    // Runtime
    CONTROL_IDLE,
    CONTROL_SWEEP_THETA_COARSE,
    CONTROL_SWEEP_THETA_FINE,
    CONTROL_SWEEP_RADIUS,
    CONTROL_CHARGING,
} ControlState;

typedef struct
{
    int32_t thetaCounts[THETA_COARSE_STEP_MAX];

    uint32_t chargeDetTimeout;
    uint32_t chargeVote;
    ControlState state;

    int16_t thetaSteps;
    int16_t radiusSteps;
    uint8_t thetaAdj;      // Theta adjustment count during radius sweep
    int8_t thetaAdjClamp;  // Hit one of the edges in theta adjust, sign is direction to keep stepping in
    uint8_t nearPhone     : 1;
    uint8_t enterFine     : 1; // Radius sweep switches from coarse to fine mode
    uint8_t sweepDir      : 1;
    uint8_t reserved      : 5;
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
                // Reset
                fsm->chargeDetTimeout = 0;
                fsm->chargeVote = 0;
                fsm->thetaAdj = 0;
                fsm->nearPhone = false;
                fsm->enterFine = false;
                fsm->sweepDir = 0;
                fsm->thetaAdjClamp = 0;

                statusSet(STATUS_CAL_0);
                fsm->state = CONTROL_CAL_MOT_0;
            }
            break;
        }

        case CONTROL_CAL_MOT_0:
        {
            if (motorCalibrate(0)) {
                fsm->radiusSteps = 0;

                statusSet(STATUS_CAL_1);
                fsm->state = CONTROL_CAL_MOT_1;
            }
            break;
        }

        case CONTROL_CAL_MOT_1:
        {
            if (motorCalibrate(1)) {
                fsm->thetaSteps = 0;

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
                    fsm->state = CONTROL_SWEEP_THETA_COARSE;
                }
            }
            break;
        }

        // Search for phone
        case CONTROL_SWEEP_THETA_COARSE:
        {
            if (motorReady(THETA_MOT)) {
                int32_t counts;
                if (soscCounts(&counts, 1)) { // Only DAC1 and SOSC1 can be used at the same time
                    // Measure out frequency counts at all coarse step positions
                    fsm->thetaCounts[fsm->thetaSteps / THETA_COARSE_STEP] = counts;
                    if (!stepTheta(THETA_COARSE_STEP)) {
                        // Find the maximum in the counts, and do fine search around there
                        // Also find the minimum
                        int32_t maxIdx = 0;
                        int32_t maxVal = 0;
                        int32_t minVal = 0x7FFFFFFF;
                        for (uint32_t i = 0; i < THETA_COARSE_STEP_MAX; ++i) {
                            if (fsm->thetaCounts[i] > maxVal) {
                                maxIdx = i;
                                maxVal = fsm->thetaCounts[i];
                            }
                            else if (fsm->thetaCounts[i] < minVal) {
                                minVal = fsm->thetaCounts[i];
                            }
                        }

                        if ((maxVal - minVal) < 40) {
                            // Don't continue sweep if there actually was no phone
                            motorDisable();
                            fsm->state = CONTROL_CAL_ENTER;
                        }
                        else {
                            // Step back an additional step to do fine serach
                            int32_t startIdx = maxIdx - 1;
                            stepTheta(-THETA_COARSE_STEP * (THETA_COARSE_STEP_MAX - startIdx));

                            // Use index 0 to store the counts to compare against in fine search
                            fsm->thetaCounts[0] = 0;

                            fsm->state = CONTROL_SWEEP_THETA_FINE;
                        }
                    }
                }
            }
            break;
        }

        case CONTROL_SWEEP_THETA_FINE:
        {
            if (motorReady(THETA_MOT)) {
                int32_t counts;
                if (soscCounts(&counts, 1)) { // Only DAC1 and SOSC1 can be used at the same time
                    // Find the maximum peak
                    if (fsm->thetaCounts[0] > 0 && counts < fsm->thetaCounts[0]) {
                        // Take a few steps backwards since the charging coil and the sosc coil are offset
                        stepTheta(-10);

                        fsm->state = CONTROL_SWEEP_RADIUS;
                    }
                    else {
                        stepTheta(1);
                    }

                    fsm->thetaCounts[0] = counts;
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

                        fsm->chargeDetTimeout = 0;
                        fsm->nearPhone = true;
                        fsm->enterFine = true; // Wait for timeout first before counting charge votes
                    }
                    else if (fsm->nearPhone && !fsm->enterFine) {
                        ++fsm->chargeVote;
                        if (fsm->chargeVote > CHARGE_DET_VOTE) { // Charging light needs to remain on, not flashing
                            fsm->chargeVote = 0;

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
                        // Zig-zag pattern to follow
                        // When can't zig-zag, keep going other way
                        // | | | | | | | |
                        //   <-
                        //   --->
                        //  <----
                        //  ------>
                        //         ->
                        //            ->
                        if (fsm->thetaAdj >= THETA_ADJUST_MAX) {
                            // Give up
                            motorDisable();
                            fsm->state = CONTROL_CAL_ENTER;
                        }
                        else {
                            int32_t step = THETA_ADJUST_STEP * (fsm->thetaAdj + 1) * stepSign;
                            int32_t newThetaStep = fsm->thetaSteps + step;

                            if (fsm->thetaAdjClamp != 0) {
                                // Keep going the other way
                                stepTheta(THETA_ADJUST_STEP * fsm->thetaAdjClamp);
                            }
                            else if (newThetaStep < 0 || newThetaStep > THETA_SWEEP_STEP_MAX) {
                                // Can't keep stepping in this direction, go the other way
                                stepTheta(THETA_ADJUST_STEP * -1 * stepSign);
                                fsm->thetaAdjClamp = -1 * stepSign;
                            }
                            else {
                                stepTheta(step);
                            }

                            ++fsm->thetaAdj;
                        }

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

        default:
        {
            fsm->state = CONTROL_CAL_ENTER;
            break;
        }
    }
}
