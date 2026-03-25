#ifdef PICO2_W
#include <pico/cyw43_arch.h>
#endif
#include <hardware/adc.h>

#include "bt.h"
#include "dac.h"
#include "control.h"
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

    // Main loop
    // In this loop the Bluetooth events will run
    while (1) {
        statusUpdate();
        dacUpdate();
        motorUpdate();

        runControl();
    }
    return 0;
}