#include <hardware/gpio.h>

#include "motor.h"
#include "hw_config.h"

enum
{
    MOT_COUNT = 2,
    MOT_STARTUP_SCALE   = 15,  // Wait count scaled by this when moving from rest
    MOT_STEP_HOLD_COUNT = 16,  // Amount of FSM updates to hold step output high
    MOT_STEP_WAIT_COUNT = 256, // Amount of FSM updates to before next step input
};

typedef enum
{
    MOT_STEP = 0,
    MOT_HOLD,    // Hold step output high
    MOT_BACKOFF, // Wait for motor to move
} MotorState;

typedef struct
{
    uint32_t waitCount;
    int32_t target;
    int32_t position;     // Offset from home position
    MotorState state;
    uint8_t startup  : 4; // Aids in motor acceleration from rest, wait count scaled by 2^startup
    uint8_t active   : 1; // Motor spinning
    uint8_t dir      : 1;
    uint8_t reserved : 2;
} Motor;

typedef enum
{
    MOT_FSM_INIT = 0,
    MOT_FSM_UPDATE,
} MotorFsmState;

typedef struct
{
    Motor motor[2];
    MotorFsmState state;
} MotorFsm;

static MotorFsm motorFsm;
static uint8_t motDirGpio[MOT_COUNT] = {GPIO_MOT_DIR_0, GPIO_MOT_DIR_1};
static uint8_t motStepGpio[MOT_COUNT] = {GPIO_MOT_STEP_0, GPIO_MOT_STEP_1};

bool motorReady(uint8_t motIdx) {
    return motorFsm.motor[motIdx].target == 0;
}

void motorMove(uint8_t motIdx, int32_t steps) {
    motorFsm.motor[motIdx].target = steps;
}

bool motorMoveHome(uint8_t motIdx) {
    int32_t pos = motorFsm.motor[motIdx].position;
    motorFsm.motor[motIdx].target = -1 * pos;
    return pos == 0;
}

void motorUpdate(void) {
    MotorFsm* fsm = &motorFsm;
    switch (fsm->state) {
        case MOT_FSM_INIT:
        {
            // All motor GPIOs are output
            uint32_t gpioMask =
                (1 << GPIO_MOT_RST) |
                (1 << GPIO_MOT_EN) |
                (1 << GPIO_MOT_SLP_0) | // Motor 0
                (1 << GPIO_MOT_MODE_0) |
                (1 << GPIO_MOT_MS3_0) |
                (1 << GPIO_MOT_MS2_0) |
                (1 << GPIO_MOT_MS1_0) |
                (1 << GPIO_MOT_STEP_0) |
                (1 << GPIO_MOT_DIR_0) |
                (1 << GPIO_MOT_SLP_1) | // Motor 1
                (1 << GPIO_MOT_MODE_1) |
                (1 << GPIO_MOT_MS3_1) |
                (1 << GPIO_MOT_MS2_1) |
                (1 << GPIO_MOT_MS1_1) |
                (1 << GPIO_MOT_STEP_1) |
                (1 << GPIO_MOT_DIR_1);
            gpio_init_mask(gpioMask);
            gpio_set_dir_out_masked(gpioMask);

            gpioMask =
                (1 << GPIO_MOT_RST)    | // Disable reset
                (1 << GPIO_MOT_SLP_0)  |
                (1 << GPIO_MOT_SLP_1)  |
                (1 << GPIO_MOT_MODE_0) | // Use automatic mixed decay mode
                (1 << GPIO_MOT_MODE_1);
            gpio_set_mask(gpioMask);

            gpioMask =
                (1 << GPIO_MOT_EN)     | // Active low, enable output
                (1 << GPIO_MOT_MS3_0)  | // Use full steps
                (1 << GPIO_MOT_MS2_0)  |
                (1 << GPIO_MOT_MS1_0)  |
                (1 << GPIO_MOT_MS3_1)  |
                (1 << GPIO_MOT_MS2_1)  |
                (1 << GPIO_MOT_MS1_1)  |
                (1 << GPIO_MOT_STEP_0) |
                (1 << GPIO_MOT_STEP_1) |
                (1 << GPIO_MOT_DIR_0)  |
                (1 << GPIO_MOT_DIR_1);
            gpio_clr_mask(gpioMask);

            fsm->state = MOT_FSM_UPDATE;
            break;
        }

        case MOT_FSM_UPDATE:
        {
            // Can update both motors at same time, since they are independent
            for (uint8_t motIdx = 0; motIdx < MOT_COUNT; ++motIdx) {
                Motor* motor = &fsm->motor[motIdx];
                switch (motor->state) {
                    case MOT_STEP:
                    {
                        if (motor->target != 0) {
                            // Change direction / motor currently stopped
                            bool targetDir = motor->target < 0;
                            if (!motor->active || motor->dir != targetDir) {
                                gpio_put(motDirGpio[motIdx], targetDir);

                                motor->startup = MOT_STARTUP_SCALE;
                                motor->active = true;
                                motor->dir = targetDir;
                            }

                            // Update tracking
                            if (targetDir) {
                                ++motor->target;
                                --motor->position;
                            }
                            else {
                                --motor->target;
                                ++motor->position;
                            }

                            uint32_t mask = 1 << motStepGpio[motIdx];
                            gpio_set_mask(mask);

                            motor->waitCount = MOT_STEP_HOLD_COUNT;
                            motor->state = MOT_HOLD;
                        }
                        else {
                            motor->active = false;
                        }
                        break;
                    }

                    case MOT_HOLD:
                    {
                        // Hold step output high
                        --motor->waitCount;
                        if (motor->waitCount == 0) {
                            uint32_t mask = 1 << motStepGpio[motIdx];
                            gpio_clr_mask(mask);

                            motor->waitCount = (uint32_t)MOT_STEP_WAIT_COUNT * (1 + motor->startup);
                            if (motor->startup > 0) {
                                --motor->startup;
                            }

                            motor->state = MOT_BACKOFF;
                        }
                        break;
                    }

                    case MOT_BACKOFF:
                    {
                        // Wait for motor to move before next step input
                        --motor->waitCount;
                        if (motor->waitCount == 0) {
                            motor->state = MOT_STEP;
                        }
                        break;
                    }

                    default:
                    {
                        motor->state = MOT_STEP;
                        break;
                    }
                }
            }
            break;
        }

        default:
        {
            fsm->state = MOT_FSM_INIT;
            break;
        }
    }
}