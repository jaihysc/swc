#include <hardware/gpio.h>
#include <hardware/timer.h>

#include "hw_config.h"
#include "status.h"

// Status LED control

// Bits determine LED periodic pattern
// Read LSB to MSB, 1=blink, 0=off
// Always blinks once at start of pattern
static const uint8_t statusMapping[] =
{
    0b00111000, // STATUS_BT_INIT
    0b01111000, // STATUS_BT_ERR
    0b00000010, // STATUS_CAL_0
    0b00000110, // STATUS_CAL_1
    0b00000000, // STATUS_IDLE
    0b00000001, // STATUS_SWEEP_THETA
    0b00000011, // STATUS_SWEEP_RADIUS
    0b00000111, // STATUS_CHARGING
    0b00000101, // STATUS_RESET_THETA
    0b00001101, // STATUS_RESET_RADIUS
};


enum
{
    LED_ON_TIME    = 100000,                 // Time LED is on if pattern at that bit is 1 [us]
    LED_BIT_TIME   = 200000,                 // Time LED spends for one pattern bit [us]
    LED_CYCLE_TIME = 1000000 + LED_BIT_TIME, // Time (plus bit time) before repeating pattern [us]
};

typedef enum
{
    LED_INIT = 0,
    LED_UPDATE,
    LED_ON_WAIT,
    LED_OFF_WAIT,
    LED_CYCLE_WAIT,
} LedState;

typedef struct
{
    uint64_t startTime;
    StatusCode nextStatus;
    StatusCode currentStatus;
    int8_t position; // Pattern bit index
    LedState state;
} Led;

static Led led;

void statusSet(StatusCode status) {
    led.nextStatus = status;
}

void statusUpdate() {
    Led* fsm = &led;

    // int32_t big enough to hold elapsed time
    int32_t elapsedTime = (int32_t)(time_us_64() - fsm->startTime);
    switch (fsm->state) {
        case LED_INIT:
        {
            gpio_init(GPIO_STATUS_LED);
            gpio_set_dir(GPIO_STATUS_LED, GPIO_OUT);

            fsm->state = LED_UPDATE;
            break;
        }

        case LED_UPDATE:
        {
            bool ledOn = true;
            if (fsm->position >= 0) { // If position == -1, always blink to mark start of new pattern
                ledOn = (statusMapping[fsm->currentStatus] >> fsm->position) & 1;
            }

            if (ledOn) {
                gpio_put(GPIO_STATUS_LED, true);
                fsm->state = LED_ON_WAIT;
            }
            else {
                fsm->state = LED_OFF_WAIT;
            }

            fsm->startTime = time_us_64();
            break;
        }

        case LED_ON_WAIT:
        {
            if (elapsedTime >= LED_ON_TIME) {
                gpio_put(GPIO_STATUS_LED, false);
                fsm->state = LED_OFF_WAIT;
            }
            break;
        }

        case LED_OFF_WAIT:
        {
            if (elapsedTime >= LED_BIT_TIME) {
                ++fsm->position;
                if (fsm->position >= 8) {
                    fsm->position = -1;  // extra position to mark start of new pattern
                    fsm->state = LED_CYCLE_WAIT;
                }
                else {
                    fsm->state = LED_INIT;
                }
            }
            break;
        }

        case LED_CYCLE_WAIT:
        {
            if (elapsedTime >= LED_CYCLE_TIME) {
                fsm->currentStatus = fsm->nextStatus;
                fsm->state = LED_INIT;
            }
            break;
        }

        default:
        {
            fsm->state = LED_INIT;
            break;
        }
    }
}