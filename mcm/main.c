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
    THETA_SWEEP_STEPS  = 5000, // TBD Steps for full rotation
    RADIUS_SWEEP_STEPS = 1000  // TBD steps for full radius
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

    dacSet(RADIUS_MOT, 498); // 0.4 V
    dacSet(THETA_MOT, 498);

    // Main loop
    // In this loop the Bluetooth events will run
    //
    // Four main states
    // Cal   -> Idle
    //  ^        v
    // Reset <- Charging
    ControlState controlState = CONTROL_CAL_ENTER;
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
                if (soscDetect(&detected, 0)) {
                    if (detected) {
                        motorMove(THETA_MOT, THETA_SWEEP_STEPS); // Sweep one rotation

                        statusSet(STATUS_SWEEP_THETA);
                        controlState = CONTROL_SWEEP_THETA_ENTER;
                    }
                }
                break;
            }

            // Search for phone
            case CONTROL_SWEEP_THETA_ENTER:
            {
                if (motorReady(0)) { // FIXME Test only state
                    motorMoveHome(0);
                }
                break;
            }

            case CONTROL_SWEEP_THETA:
            {
                bool detected;
                if (soscDetect(&detected, 1)) { // Only DAC1 and SOSC1 can be used at the same time
                    if (detected) {
                        motorMove(THETA_MOT, 0); // Stop motor
                        // dacEnable(THETA_MOT, false);

                        // // Turn on DAC and motor
                        // dacEnable(RADIUS_MOT, true);
                        // controlState = CONTROL_SWEEP_RADIUS_ENTER;
                        motorMove(RADIUS_MOT, RADIUS_SWEEP_STEPS); // Sweep radius

                        statusSet(STATUS_SWEEP_RADIUS);
                        controlState = CONTROL_SWEEP_RADIUS;
                    }
                    else if (motorReady(THETA_MOT)) {
                        // Sweep in other direction
                        if (motorMoveHome(THETA_MOT)) {
                            // If already at start, sweep again one rotation
                            motorMove(THETA_MOT, THETA_SWEEP_STEPS);
                        }
                    }
                }
                break;
            }

            // case CONTROL_SWEEP_RADIUS_ENTER:
            // {
            //     if (!dacActive(THETA_MOT) && dacReady(RADIUS_MOT)) {
            //         motorMove(RADIUS_MOT, RADIUS_SWEEP_STEPS); // Sweep radius

            //         statusSet(STATUS_SWEEP_RADIUS);
            //         controlState = CONTROL_SWEEP_RADIUS;
            //     }
            // }

            case CONTROL_SWEEP_RADIUS:
            {
                if (gpio_get(GPIO_CHARGING)) {
                    motorMove(RADIUS_MOT, 0); // Stop motor
                    dacEnable(RADIUS_MOT, false);

                    statusSet(STATUS_CHARGING);
                    controlState = CONTROL_CHARGING;
                }
                else if (motorReady(RADIUS_MOT)) {
                    // Sweep in other direction
                    if (motorMoveHome(RADIUS_MOT)) {
                        // If already at start, sweep again radius
                        motorMove(RADIUS_MOT, RADIUS_SWEEP_STEPS);
                    }
                }
                break;
            }

            // Charging
            case CONTROL_CHARGING:
            {
                if (!gpio_get(GPIO_CHARGING)) {
                    // Rotate back to home
                    dacEnable(RADIUS_MOT, true);
                    controlState = CONTROL_RESET_RADIUS_ENTER;
                }
                break;
            }

            // Reset
            case CONTROL_RESET_RADIUS_ENTER:
            {
                if (dacReady(RADIUS_MOT)) {
                    motorMoveHome(RADIUS_MOT);

                    statusSet(STATUS_RESET_RADIUS);
                    controlState = CONTROL_RESET_RADIUS;
                }
                break;
            }

            case CONTROL_RESET_RADIUS:
            {
                if (motorReady(RADIUS_MOT)) {
                    dacEnable(RADIUS_MOT, false);

                    // Now take theta to home position
                    dacEnable(THETA_MOT, true);
                    controlState = CONTROL_RESET_THETA_ENTER;
                }
                break;
            }

            case CONTROL_RESET_THETA_ENTER:
            {
                if (dacReady(THETA_MOT)) {
                    motorMoveHome(THETA_MOT);

                    statusSet(STATUS_RESET_THETA);
                    controlState = CONTROL_RESET_THETA;
                }
                break;
            }

            case CONTROL_RESET_THETA:
            {
                if (motorReady(THETA_MOT)) {
                    dacEnable(THETA_MOT, false);
                    controlState = CONTROL_CAL_ENTER;
                }
                break;
            }

            // Test modes
            case CONTROL_TEST_MOTOR_0:
            {
                if (dacReady(0)) {
                    motorMove(0, 100); // This will be repeatedly called to spin motor forever
                }
                break;
            }

            case CONTROL_TEST_MOTOR_1:
            {
                if (dacReady(1)) {
                    motorMove(1, 100);
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