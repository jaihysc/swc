#ifdef PICO2_W
#include <pico/cyw43_arch.h>
#endif
#include <hardware/adc.h>

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


    // Initialize GPIO
    // Some GPIO settings are part of FSMs in main loop
    gpio_init(GPIO_STATUS_LED);
    gpio_set_dir(GPIO_STATUS_LED, GPIO_OUT);

    gpio_init(15);
    gpio_set_dir(15, GPIO_IN);
    gpio_init(12);
    gpio_set_dir(12, GPIO_IN);

    adc_init();

    btInit();


    // Test
    dacSet(0, 3724); // 3/3.3 * 2^12
    dacSet(1, 3475);


    // Main loop
    // In this loop the Bluetooth events will run
    while (1) {
        dacUpdate();
        soscUpdate();
    }
    return 0;
}