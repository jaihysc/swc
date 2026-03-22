#ifdef PICO2_W
#include <pico/cyw43_arch.h>
#endif
#include <hardware/adc.h>

#include "control.h"
#include "bt.h"
#include "dac.h"
#include "motor.h"
#include "hw_config.h"
#include "sosc.h"
#include "status.h"

enum
{
    THETA_MOT = 1, // Motor/DAC for theta sweep, must be 1 since only DAC1 can run with SOSC1 when sweeping
    RADIUS_MOT = 0,
    // THETA_SWEEP_STEPS  = 6000, // Steps for full rotation
    // RADIUS_SWEEP_STEPS = 1200  // steps for full radius
    THETA_SWEEP_STEP_MAX = 40,
    THETA_SWEEP_STEP  = 150,

    RADIUS_SWEEP_STEP_MAX = 30,
    RADIUS_SWEEP_STEP = 40,

    // THETA_SWEEP_STEP_MAX  = 150,
    // RADIUS_SWEEP_STEP_MAX = 40
};

int main() {
#ifdef PICO2_W
    // Initialize CYW43 driver architecture (will enable BT if/because CYW43_ENABLE_BLUETOOTH == 1)
    if (cyw43_arch_init()) {
        return 1; // Can't use LED if this fails
    }
#endif

    // Initialize hardware
    // Some initializations settings are part of FSMs in main loop

    gpio_init(GPIO_CHARGING);
    gpio_set_dir(GPIO_CHARGING, GPIO_IN);

    adc_init();
    btInit();

    soscInit();

    // Main loop
    // In this loop the Bluetooth events will run
    //
    // Four main states
    // Cal   -> Idle
    //  ^        v
    // Reset <- Charging
    ControlState controlState = CONTROL_CAL_ENTER;

    // Sweep steps / current direction
    int16_t threshAdjust = 0;
    int8_t sweepSteps = 0;
    int8_t sweepDir = 1;
    while (1) {
        statusUpdate();
        dacUpdate();
        motorUpdate();

        switch (controlState) {
            // Calibrate
            case CONTROL_CAL_ENTER:
            {
                if (!dacActive(0) && !dacActive(1)) {
                    controlState = CONTROL_CAL_MOT_0;
                }
                break;
            }

            case CONTROL_CAL_MOT_0:
            {
                if (motorCalibrate(0)) {
                    controlState = CONTROL_CAL_MOT_1;
                }
                break;
            }

            case CONTROL_CAL_MOT_1:
            {
                if (motorCalibrate(1)) {
                    statusSet(STATUS_CAL_1);
                    controlState = CONTROL_CAL_SOSC_1;
                }
                break;
            }

            case CONTROL_CAL_SOSC_1: // First calibrate SOSC1, then SOSC0, so SOSC0 can remain on for idle
            {
                if (soscCalibrate(1)) {
                    statusSet(STATUS_CAL_1);
                    controlState = CONTROL_CAL_SOSC_0;
                }
                break;
            }

            case CONTROL_CAL_SOSC_0:
            {
                if (soscCalibrate(0)) {
                    statusSet(STATUS_IDLE);
                    controlState = CONTROL_IDLE;
                }
                break;
            }

            // Idle
            case CONTROL_IDLE:
            {
                bool detected;
                if (soscDetect(&detected, 0, 0)) {
                    if (detected) {
                        statusSet(STATUS_SWEEP_THETA);

                        threshAdjust = 0;
                        sweepSteps = 0;
                        sweepDir = 1;
                        controlState = CONTROL_SWEEP_THETA;
                    }
                }
                break;
            }

            // Search for phone
            case CONTROL_SWEEP_THETA:
            {
                bool detected;
                if (motorReady(THETA_MOT)) {
                    if (soscDetect(&detected, 1, threshAdjust)) { // Only DAC1 and SOSC1 can be used at the same time
                        if (detected) {
                            statusSet(STATUS_SWEEP_RADIUS);
                            controlState = CONTROL_SWEEP_RADIUS;
                        }
                        else {
                            if (sweepDir > 0) {
                                motorMove(THETA_MOT, THETA_SWEEP_STEP);
                                ++sweepSteps;
                                if (sweepSteps >= THETA_SWEEP_STEP_MAX) {
                                    // At the end, next step try sweeping backwards
                                    // Lower the threshold
                                    sweepDir = -1;
                                    threshAdjust -= 16;
                                }
                            }
                            else {
                                motorMove(THETA_MOT, -THETA_SWEEP_STEP);
                                --sweepSteps;
                                if (sweepSteps == 0) {
                                    sweepDir = 1;
                                    threshAdjust -= 16;
                                }
                            }
                        }
                    }
                }
                break;
            }

            case CONTROL_SWEEP_RADIUS:
            {
                // TODO
                // if (gpio_get(GPIO_CHARGING)) {
                //     motorMove(RADIUS_MOT, 0); // Stop motor
                //     dacEnable(RADIUS_MOT, false);

                //     statusSet(STATUS_CHARGING);
                //     controlState = CONTROL_CHARGING;
                // }
                // else if (motorReady(RADIUS_MOT)) {
                //     // Sweep in other direction
                //     if (motorMoveHome(RADIUS_MOT)) {
                //         // If already at start, sweep again radius
                //         motorMove(RADIUS_MOT, RADIUS_SWEEP_STEPS);
                //     }
                // }
                break;
            }

            // Charging
            case CONTROL_CHARGING:
            {
                if (!gpio_get(GPIO_CHARGING)) {
                    // Rotate back to home
                    dacEnable(RADIUS_MOT, true);
                    controlState = CONTROL_CAL_ENTER;
                }
                break;
            }

            default:
            {
                controlState = CONTROL_CAL_ENTER;
                break;
            }
        }
    }
    return 0;
}