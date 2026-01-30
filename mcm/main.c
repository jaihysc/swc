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
    // gpio_init(15);
    // gpio_set_dir(15, GPIO_IN);
    // gpio_init(12);
    // gpio_set_dir(12, GPIO_IN);

    adc_init();

    btInit();
    soscInit();

    // Main loop
    // In this loop the Bluetooth events will run
    ControlState controlState = CONTROL_IDLE_ENTER; //CONTROL_CAL_ENTER;
    dacSet(0, 498); // 0.4 V
    dacEnable(0, true);
    while (1) {
        statusUpdate();
        dacUpdate();
        motorUpdate();

        switch (controlState) {
            // Calibrate
            case CONTROL_CAL_ENTER:
            {
                statusSet(STATUS_CAL_0);
                controlState = CONTROL_CAL_0;
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
                    controlState = CONTROL_IDLE_ENTER;
                }
                break;
            }

            // Runtime
            case CONTROL_IDLE_ENTER:
            {
                // if (!dacActive(1)) {
                if (motorReady(0)) {
                    motorMove(0, 5000); // Test

                    statusSet(STATUS_IDLE);
                    controlState = CONTROL_IDLE;
                }
                break;
            }

            case CONTROL_IDLE:
            {
                if (motorReady(0)) {
                    motorMove(0, -5000); // Test
                    controlState = CONTROL_IDLE_ENTER;
                }
                // bool detected;
                // if (soscDetect(&detected, 0)) {
                //     if (detected) {
                //         dacEnable(0, false);
                //         controlState = CONTROL_SWEEP_THETA_ENTER;
                //     }
                // }
                break;
            }

            case CONTROL_SWEEP_THETA_ENTER:
            {
                // if (!dacActive(0)) {
                //     dacEnable(1, true);

                //     statusSet(STATUS_SWEEP_THETA);
                //     controlState = CONTROL_SWEEP_THETA;
                // }
                break;
            }

            case CONTROL_SWEEP_THETA:
            {
                bool detected;
                if (soscDetect(&detected, 1)) {
                    if (detected) {
                        dacEnable(1, false);
                        controlState = CONTROL_CAL_ENTER;
                    }
                }
                break;
            }

            case CONTROL_SWEEP_RADIUS_ENTER:
            {
                break;
            }

            case CONTROL_SWEEP_RADIUS:
            {
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