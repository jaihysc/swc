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
    RADIUS_SWEEP_STEP_MAX     = 80,
    RADIUS_SWEEP_STEP         = 15,

    // Fine search around the index where max was found
    // Coarse steps |     |     |     |     |
    // Fine steps    <--------->
    //
    // The fine sweep range (width) is based on how spread the coarse sweep data is
    //                                | <-- Max in coarse sweep
    // Coarse sweep steps |     |     |     |     |
    //                     <--Width--> <--Width-->

    MAX_SWEEP_RETRY           = 2,

    THETA_COARSE_STEP         = 20, // Coarse steps are 20x the theta step, which is 24x the minimum step size
    THETA_COARSE_STEP_MAX     = THETA_SWEEP_STEP_MAX / THETA_COARSE_STEP,

    THETA_FINE_WIDTH_MAX      = 4,  // Coarse steps around max found in coarse sweep
    THETA_FINE_WIDTH_COUNT    = 16, // For counts near maximum, minimum count difference from maximum to expand width
    THETA_FINE_STEP           = 2,
    THETA_FINE_STEP_MAX       = 2 * THETA_FINE_WIDTH_MAX * THETA_COARSE_STEP / THETA_FINE_STEP,

    THETA_ADJUST_STEP         = 2, // When radius sweep couldn't find phone
    THETA_ADJUST_MAX          = 1, // This needs to be odd, since the charger needs to come back the front

    CHARGE_DET_VOTE           = 1 << 17, // FSM iterations to declare charging
    CHARGE_DET_TIMEOUT        = (1 << 17) + (1 << 16),  // FSM iterations to check for charging
};

typedef enum
{
    // Enter states wait for hardware in previous state to be disabled
    // Four main states
    // Cal   -> Idle
    //  ^        v
    // Reset <- Charging

    // Startup calibration
    CONTROL_INIT = 0,
    CONTROL_CAL_MOT_0,
    CONTROL_CAL_MOT_1,

    // Runtime
    CONTROL_IDLE,
    CONTROL_SWEEP_THETA_COARSE,
    CONTROL_PICK_PIVOT,
    CONTROL_SWEEP_THETA_FINE,
    CONTROL_SWEEP_RADIUS,
    CONTROL_CHARGING,
} ControlState;

typedef struct
{
    int32_t thetaCountsCoarse[THETA_COARSE_STEP_MAX + 1]; // n steps, but n + 1 measurements (start and end of step)
    int32_t thetaCountsFine[THETA_FINE_STEP_MAX + 1];

    uint32_t chargeDetTimeout;
    uint32_t chargeVote;
    ControlState state;

    int16_t thetaSteps;
    int16_t radiusSteps;

    uint8_t retryCount;
    uint8_t coarseIdx;         // Index of next measurement
    uint8_t fineIdx;

    uint8_t fineSweepWidth;    // Calculated during coarse sweep, used by fine sweep
    uint8_t thetaAdjCount;     // Theta adjustment count during radius sweep
    int8_t thetaAdjClamp;      // Hit one of the edges in theta adjust, sign is direction to keep stepping in
    uint8_t nearPhone     : 1;
    uint8_t enterFine     : 1; // Radius sweep switches from coarse to fine mode
    uint8_t sweepDir      : 1;
    uint8_t reserved      : 5;
} Control;

int32_t coarseCorrFact[THETA_COARSE_STEP_MAX + 1] = { // Corrects offsets in measured counts at coarse positions
   -20,
   -11,
    -7,
     0,
    -2,
    -2,
    -3,
    -4,
    -5,
   -10,
   -12,
   -17,
   -22,
   -29
};

Control control;

// Tries to move theta by provided number of THETA STEPS, true if moved
static bool stepTheta(int16_t thetaStep) {
    int16_t newStep = control.thetaSteps + thetaStep;

    // Clamp
    if (newStep < 0) {
        thetaStep = -1 * control.thetaSteps;
    }
    else if (newStep > THETA_SWEEP_STEP_MAX) {
        thetaStep = THETA_SWEEP_STEP_MAX - control.thetaSteps;
    }

    if (thetaStep == 0) {
        return false; // Can't move
    }
    else {
        motorMove(THETA_MOT, THETA_SWEEP_STEP * thetaStep);
        control.thetaSteps += thetaStep;
        return true;
    }
}

// Moves theta to provided THETA STEP position
static void setTheta(int16_t thetaPos) {
    // Clamp
    if (thetaPos < 0) {
        thetaPos = 0;
    }
    else if (thetaPos > THETA_SWEEP_STEP_MAX) {
        thetaPos = THETA_SWEEP_STEP_MAX;
    }

    motorMove(THETA_MOT, (THETA_SWEEP_STEP * thetaPos) - motorPosition(THETA_MOT));
    control.thetaSteps = thetaPos;
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
        case CONTROL_INIT:
        {
            soscPowerup(0);

            statusSet(STATUS_CAL_0);
            fsm->state = CONTROL_CAL_MOT_0;
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

                    // Reset coarse sweep variables
                    fsm->coarseIdx = 0;
                    fsm->state = CONTROL_SWEEP_THETA_COARSE;
                }
            }
            break;
        }

        // Search for phone
        case CONTROL_SWEEP_THETA_COARSE:
        {
            // Coarse sweep runs in the forward direction

            if (motorReady(THETA_MOT)) {
                int32_t counts;
                if (soscCounts(&counts, 1)) { // Only DAC1 and SOSC1 can be used at the same time
                    // Measure out frequency counts at all coarse step positions
                    fsm->thetaCountsCoarse[fsm->coarseIdx] = counts + coarseCorrFact[fsm->coarseIdx];

                    bool doneSweep = fsm->coarseIdx >= THETA_COARSE_STEP_MAX;
                    bool canStep = false;
                    if (!doneSweep) {
                        canStep = stepTheta(THETA_COARSE_STEP);
                    }

                    ++fsm->coarseIdx;

                    // At the end, or took the required number of steep steps
                    if (!canStep || doneSweep) {
                        // The calculated counts and coarseIdx is used in pick pivot
                        fsm->state = CONTROL_PICK_PIVOT;
                    }
                }
            }
            break;
        }

        case CONTROL_PICK_PIVOT:
        {
            // Find the maximum in the counts (pivot), and do fine search around there
            int32_t maxIdx = 0;
            int32_t maxVal = 0;
            for (int32_t i = 0; i < fsm->coarseIdx; ++i) {
                if (fsm->thetaCountsCoarse[i] > maxVal) {
                    maxIdx = i;
                    maxVal = fsm->thetaCountsCoarse[i];
                }
            }

            // Clear the maxIdx and its neighbors
            // When picking pivot again, it will pick the next smallest pivot
            fsm->thetaCountsCoarse[maxIdx] = 0;

            // Calculate width of fine sweep
            uint32_t leftWidth = 0;
            for (int32_t i = maxIdx - 1; i >= 0; --i) {
                if ((maxVal - fsm->thetaCountsCoarse[i]) > THETA_FINE_WIDTH_COUNT) {
                    break;
                }
                fsm->thetaCountsCoarse[i] = 0;
                ++leftWidth;
                if (leftWidth >= THETA_FINE_WIDTH_MAX) {
                    break;
                }
            }

            uint32_t rightWidth = 0;
            for (int32_t i = maxIdx + 1; i < fsm->coarseIdx; ++i) {
                if ((maxVal - fsm->thetaCountsCoarse[i]) > THETA_FINE_WIDTH_COUNT) {
                    break;
                }
                fsm->thetaCountsCoarse[i] = 0;
                ++rightWidth;
                if (rightWidth >= THETA_FINE_WIDTH_MAX) {
                    break;
                }
            }

            // Take maximum and clamp (clamp for maximum is done inside for loop)
            fsm->fineSweepWidth = (leftWidth > rightWidth) ? leftWidth : rightWidth;
            if (fsm->fineSweepWidth <= 0) {
                fsm->fineSweepWidth = 1;
            }

            // Step to start of fine sweep
            int32_t startIdx = maxIdx + fsm->fineSweepWidth;
            setTheta(startIdx * THETA_COARSE_STEP);

            // Reset variables for fine sweep
            fsm->fineIdx = 0;

            fsm->state = CONTROL_SWEEP_THETA_FINE;
            break;
        }

        case CONTROL_SWEEP_THETA_FINE:
        {
            // Fine sweep runs in the reverse direction

            if (motorReady(THETA_MOT)) {
                int32_t counts;
                if (soscCounts(&counts, 1)) {
                    // Measure out frequency counts at all fine step positions
                    fsm->thetaCountsFine[fsm->fineIdx] = counts;

                    bool doneSweep = fsm->fineIdx >= (2 * fsm->fineSweepWidth * THETA_COARSE_STEP / THETA_FINE_STEP);
                    bool canStep = false;
                    if (!doneSweep) {
                        canStep = stepTheta(-THETA_FINE_STEP);
                    }

                    ++fsm->fineIdx;

                    if (!canStep || doneSweep) {
                        // Find the maximum in the counts, do sweep with the charger
                        int32_t maxIdx = 0;
                        int32_t maxVal = 0;
                        for (int32_t i = 0; i < fsm->fineIdx; ++i) {
                            int32_t val = fsm->thetaCountsFine[i];
                            // Greater than or equal, we want the rightmost maximum since theta adjust moves left first
                            if (val >= maxVal) {
                                maxIdx = i;
                                maxVal = val;
                            }
                        }

                        // Take extra 5 steps backwards since the charging coil and the sosc coil are offset
                        stepTheta(THETA_FINE_STEP * (fsm->fineIdx - maxIdx) - 5);

                        // Reset variables for radius sweep
                        fsm->chargeDetTimeout = 0;
                        fsm->chargeVote = 0;

                        fsm->thetaAdjCount = 0;
                        fsm->thetaAdjClamp = 0;

                        fsm->nearPhone = false;
                        fsm->enterFine = false;
                        fsm->sweepDir = 0;

                        statusSet(STATUS_SWEEP_RADIUS);
                        fsm->state = CONTROL_SWEEP_RADIUS;
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
                    // When charging light first turns on, go back and slowly step forwards
                    if (!fsm->nearPhone) {
                        stepRadius(-1 * stepSign * step);

                        fsm->chargeDetTimeout = 0;
                        fsm->nearPhone = true;
                        fsm->enterFine = true; // Wait for timeout first before counting charge votes
                    }
                    else if (fsm->nearPhone && !fsm->enterFine) {
                        ++fsm->chargeVote;
                        if (fsm->chargeVote > CHARGE_DET_VOTE) { // Charging light needs to remain on, not flashing
                            fsm->chargeVote = 0;
                            fsm->retryCount = 0;

                            motorDisable(); // Turn off motors to save power

                            statusSet(STATUS_CHARGING);
                            fsm->state = CONTROL_CHARGING;
                        }
                    }
                }
                else {
                    fsm->chargeVote = 0;
                }

                ++fsm->chargeDetTimeout;
                if (fsm->enterFine) {
                    // When first entering fine mode, need to wait extra time for charging light to clear
                    if (fsm->chargeDetTimeout > (4 * CHARGE_DET_TIMEOUT)) {
                        fsm->chargeDetTimeout = 0;
                        fsm->enterFine = false;
                    }
                }
                else if (fsm->chargeDetTimeout > CHARGE_DET_TIMEOUT) {
                    // Didn't find phone, keep sweeping
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
                        if (fsm->thetaAdjCount >= THETA_ADJUST_MAX) {
                            // Give up, try fine sweep again or return to idle
                            ++fsm->retryCount;
                            if (fsm->retryCount > MAX_SWEEP_RETRY) {
                                fsm->retryCount = 0;
                                fsm->state = CONTROL_INIT;
                            }
                            else {
                                fsm->state = CONTROL_PICK_PIVOT;
                            }
                        }
                        else {
                            int32_t step = THETA_ADJUST_STEP * (fsm->thetaAdjCount + 1) * stepSign;
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

                            ++fsm->thetaAdjCount;
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
                    fsm->state = CONTROL_INIT;
                }
            }
            else {
                fsm->chargeVote = 0;
            }
            break;
        }

        default:
        {
            fsm->state = CONTROL_INIT;
            break;
        }
    }
}
