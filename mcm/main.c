#ifdef PICO2_W
#include <pico/cyw43_arch.h>
#endif
#include <hardware/adc.h>

#include "control.h"
#include "bt.h"
#include "dac.h"
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

    // Main loop
    // In this loop the Bluetooth events will run
    ControlState controlState = CONTROL_INIT;

    while (1) {
        dacUpdate();
        soscUpdate();
        statusUpdate();

        switch (controlState) {
            // Calibrate
            case CONTROL_INIT:
            {
                statusSet(STATUS_CAL);

                soscEnable(0, true);
                controlState = CONTROL_CAL_0;
                break;
            }

            case CONTROL_CAL_0:
            {
                if (soscCalibrated(0)) {
                    soscEnable(0, false);
                    controlState = CONTROL_CAL_1_ENTER;
                }
                break;
            }

            case CONTROL_CAL_1_ENTER:
            {
                if (!soscActive(0)) {
                    soscEnable(1, true);
                    controlState = CONTROL_CAL_1;
                }
                break;
            }

            case CONTROL_CAL_1:
            {
                if (soscCalibrated(1)) {
                    soscEnable(1, false);
                    controlState = CONTROL_IDLE_ENTER;
                }
                break;
            }

            // Runtime
            case CONTROL_IDLE_ENTER:
            {
                if (!dacActive(1) && !soscActive(1)) {
                    dacEnable(0, true);
                    soscEnable(0, true);

                    statusSet(STATUS_IDLE);
                    controlState = CONTROL_IDLE;
                }
                break;
            }

            case CONTROL_IDLE:
            {
                if (soscDetected(0)) {
                    dacEnable(0, false);
                    soscEnable(0, false);
                    controlState = CONTROL_SWEEP_THETA_ENTER;
                }
                break;
            }

            case CONTROL_SWEEP_THETA_ENTER:
            {
                if (!dacActive(0) && !soscActive(0)) {
                    dacEnable(1, true);
                    soscEnable(1, true);

                    statusSet(STATUS_SWEEP_THETA);
                    controlState = CONTROL_SWEEP_THETA;
                }
                break;
            }

            case CONTROL_SWEEP_THETA:
            {
                if (soscDetected(1)) {
                    dacEnable(1, false);
                    soscEnable(1, false);
                    controlState = CONTROL_IDLE_ENTER;
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
                controlState = CONTROL_INIT;
                break;
            }
        }
    }
    return 0;
}