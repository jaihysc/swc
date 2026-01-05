#include <hardware/adc.h>
#include <hardware/pwm.h>

#include "dac.h"
#include "hw_config.h"

// 12 bit ADC allows 2^12 DAC output levels
// 2^4 fractional output levels via dithering between level and level+1

enum
{
    DAC_COUNT       = 2,
    DAC_INT_BITS    = 12,
    DAC_FRAC_BITS   = 4,
    DAC_PWM_WRAP    = 1 << DAC_INT_BITS,
    DAC_WAIT_CYCLES = 1 << 10,
    DAC_LEVEL_MIN   = 0,
    DAC_LEVEL_MAX   = (((1 << DAC_INT_BITS) - 1 - 1) << DAC_FRAC_BITS) + ((1 << DAC_FRAC_BITS) - 1),
    //                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //                Extra -1 for int code because dithering needs to use level + 1
};

typedef enum
{
    DAC_INIT = 0,
    DAC_MEAS_WAIT,
    DAC_WAIT,
} DacState;

typedef struct
{
    uint32_t fracCounter[DAC_COUNT];
    uint32_t waitCounter;
    uint16_t level[DAC_COUNT];
    uint16_t target[DAC_COUNT];
    DacState state;
    uint8_t dacIdx;
} Dac;

static Dac dac;

void dacSet(uint8_t dacIdx, uint16_t val) {
    dac.level[dacIdx] = (val << DAC_FRAC_BITS);
    dac.target[dacIdx] = val;
}

void dacUpdate(void) {
    Dac* fsm = &dac;

    // Update dithering based on frac code
    // 1 frac code step = 1 cycle on PWM, assumes fracCounter incremented every clock cycle
    // When frac code counter below frac code, use code + 1
    // When frac code counter above frac code, use code
    for (int dacIdx = 0; dacIdx < DAC_COUNT; ++dacIdx) {
        const uint16_t level = fsm->level[dacIdx];
        const uint16_t levelInt = level >> DAC_FRAC_BITS;
        const uint16_t levelFrac = level & ((1 << DAC_FRAC_BITS) - 1);

        uint32_t fracCounter = fsm->fracCounter[dacIdx];
        uint16_t chanLevel = levelInt;
        ++fracCounter;
        if (fracCounter >= (DAC_PWM_WRAP * DAC_FRAC_BITS)) {
            fracCounter = 0;
            chanLevel = levelInt + 1;
        }
        if (fracCounter >= (DAC_PWM_WRAP * levelFrac)) {
            chanLevel = levelInt;
        }
        fsm->fracCounter[dacIdx] = fracCounter;

        if (dacIdx == 0) {
            pwm_set_chan_level(PWM_SLICE_DAC_OUT_0, PWM_CHAN_DAC_OUT_0, chanLevel);
        }
        else {
            pwm_set_chan_level(PWM_SLICE_DAC_OUT_1, PWM_CHAN_DAC_OUT_1, chanLevel);
        }
    }

    // Update FSM
    switch (fsm->state) {
        case DAC_INIT:
        {
            // Configure PWM hardware
            pwm_set_wrap(PWM_SLICE_DAC_OUT_0, DAC_PWM_WRAP);
            pwm_set_wrap(PWM_SLICE_DAC_OUT_1, DAC_PWM_WRAP);

            pwm_set_enabled(PWM_SLICE_DAC_OUT_0, true);
            pwm_set_enabled(PWM_SLICE_DAC_OUT_1, true);

            fsm->state = DAC_WAIT;
            break;
        }

        case DAC_WAIT:
        {
            // Wait for output to settle
            ++fsm->waitCounter;
            if (fsm->waitCounter > DAC_WAIT_CYCLES) {
                fsm->waitCounter = 0;
                adc_select_input(ADC_DAC_SENSE_0);
                hw_set_bits(&adc_hw->cs, ADC_CS_START_ONCE_BITS);

                fsm->state = DAC_MEAS_WAIT;
            }
            break;
        }

        case DAC_MEAS_WAIT:
        {
            if (adc_hw->cs & ADC_CS_READY_BITS) {
                uint16_t senseVal = adc_hw->result; // Voltage = result / 2^12

                // Raise duty cycle if output voltage below target
                // Decrease duty cycle if output voltage above target
                const uint8_t dacIdx = fsm->dacIdx;
                uint16_t level = fsm->level[dacIdx];
                if (senseVal < fsm->target[dacIdx]) {
                    if (level < DAC_LEVEL_MAX) {
                        ++level;
                    }
                }
                else {
                    if (level > DAC_LEVEL_MIN) {
                        --level;
                    }
                }

                fsm->level[dacIdx] = level;

                if (fsm->dacIdx == 0) {
                    adc_select_input(ADC_DAC_SENSE_1);
                    hw_set_bits(&adc_hw->cs, ADC_CS_START_ONCE_BITS);
                    fsm->dacIdx = 1;
                }
                else {
                    fsm->dacIdx = 0;
                    fsm->state = DAC_WAIT;
                }
            }
            break;
        }

        default:
        {
            fsm->state = DAC_INIT;
        }
    }
}