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

    dacSet(0, 498); // 0.4 V
    dacSet(1, 498);

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
                    statusSet(STATUS_CAL_0);
                    controlState = CONTROL_CAL_0;
                }
                break;
            }

            case CONTROL_CAL_0:
            {
                if (soscCalibrate(0)) {
                    statusSet(STATUS_CAL_1);
                    controlState = CONTROL_CAL_1;
                }
                break;
            }

            case CONTROL_CAL_1:
            {
                if (soscCalibrate(1)) {
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
                        // Turn on DAC and motor 0
                        dacEnable(0, true);
                        controlState = CONTROL_SWEEP_THETA_ENTER;
                    }
                }
                break;
            }

            // Search for phone
            case CONTROL_SWEEP_THETA_ENTER:
            {
                if (dacReady(0)) {
                    // Sweep one rotation
                    motorMove(0, 5000); // <<-- How many step for one rotation ?

                    statusSet(STATUS_SWEEP_THETA);
                    controlState = CONTROL_SWEEP_THETA;
                }
                break;
            }

            case CONTROL_SWEEP_THETA:
            {
                bool detected;
                if (soscDetect(&detected, 1)) {
                    if (detected) {
                        motorMove(0, 0); // Stop motor 0
                        dacEnable(0, false);

                        // Turn on DAC and motor 1
                        dacEnable(1, true);
                        controlState = CONTROL_SWEEP_RADIUS_ENTER;
                    }
                }
                else if (motorReady(0)) {
                    // Sweep in other direction
                    if (motorMoveHome(0)) {
                        // If already at start, sweep again one rotation
                        motorMove(0, 5000); // <<-- How many step for one rotation ?
                    }
                }
                break;
            }

            case CONTROL_SWEEP_RADIUS_ENTER:
            {
                if (!dacActive(0) && dacReady(1)) {
                    // Sweep radius
                    motorMove(1, 5000); // <<-- How many step to move to one end ?

                    statusSet(STATUS_SWEEP_RADIUS);
                    controlState = CONTROL_SWEEP_RADIUS;
                }
            }

            case CONTROL_SWEEP_RADIUS:
            {
                if (gpio_get(GPIO_CHARGING)) {
                    motorMove(1, 0); // Stop motor 1
                    dacEnable(1, false);

                    statusSet(STATUS_CHARGING);
                    controlState = CONTROL_CHARGING;
                }
                else if (motorReady(1)) {
                    // Sweep in other direction
                    if (motorMoveHome(1)) {
                        // If already at start, sweep again radius
                        motorMove(1, 5000); // <<-- How many step to move to one end ?
                    }
                }
                break;
            }

            // Charging
            case CONTROL_CHARGING:
            {
                if (!gpio_get(GPIO_CHARGING)) {
                    dacEnable(1, true);
                    controlState = CONTROL_RESET_RADIUS_ENTER;
                }
                break;
            }

            // Reset
            case CONTROL_RESET_RADIUS_ENTER:
            {
                if (dacReady(1)) {
                    motorMoveHome(1);

                    statusSet(STATUS_RESET_RADIUS);
                    controlState = CONTROL_RESET_RADIUS;
                }
                break;
            }

            case CONTROL_RESET_RADIUS:
            {
                if (motorReady(1)) {
                    dacEnable(1, false);

                    dacEnable(0, true);
                    controlState = CONTROL_RESET_THETA_ENTER;
                }
                break;
            }

            case CONTROL_RESET_THETA_ENTER:
            {
                if (dacReady(0)) {
                    motorMoveHome(0);

                    statusSet(STATUS_RESET_THETA);
                    controlState = CONTROL_RESET_THETA;
                }
                break;
            }

            case CONTROL_RESET_THETA:
            {
                if (motorReady(0)) {
                    dacEnable(0, false);
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