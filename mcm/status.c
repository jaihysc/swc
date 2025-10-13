#include <pico/cyw43_arch.h>
#include "status.h"

// Bits determine LED periodic pattern
// Read MSB to LSB, 1=high, 0=low
static const uint8_t statusMapping[] =
{
    0b11101000, // STATUS_BT_INIT
    0b11101010, // STATUS_BT_ERR
    0b10000000, // STATUS_IDLE
    0b10101010, // STATUS_TRANSIT
};

static StatusCode nextStatus = STATUS_OK;
static StatusCode currentStatus = STATUS_OK;
static uint8_t position = 0;

void statusSet(StatusCode status) {
    nextStatus = status;
}

void statusUpdate() {
    if (position == 0) {
        position = 8;
        currentStatus = nextStatus; // Load new status pattern
    }
    else {
        --position;
        bool ledOn = (statusMapping[currentStatus] >> position) & 1;
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, ledOn);
    }
}