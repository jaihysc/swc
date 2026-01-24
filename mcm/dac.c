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
    DAC_WAIT_CYCLES = 1 << 6,
    DAC_LEVEL_MIN   = 0,
    DAC_LEVEL_MAX   = (((1 << DAC_INT_BITS) - 1 - 1) << DAC_FRAC_BITS) + ((1 << DAC_FRAC_BITS) - 1),
    //                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //                Extra -1 for int code because dithering needs to use level + 1
};

typedef struct
{
    uint32_t fracCounter;
    uint16_t level;
    uint16_t target;
    uint8_t enable   : 1;
    uint8_t active   : 1;
    uint8_t reserved : 6;
} Dac;

typedef enum
{
    DAC_INIT = 0,
    DAC_MEAS_INIT,
    DAC_MEAS_WAIT,
    DAC_MEAS_DONE,
    DAC_WAIT,
} DacState;

typedef struct
{
    Dac dac[DAC_COUNT];
    uint32_t waitCounter;
    DacState state;
    uint8_t dacIdx;
} DacFsm;

static DacFsm dacFsm;

void dacEnable(uint8_t dacIdx, bool enable) {
    dacFsm.dac[dacIdx].enable = enable;
}

bool dacActive(uint8_t dacIdx) {
    return dacFsm.dac[dacIdx].active;
}

void dacSet(uint8_t dacIdx, uint16_t val) {
    dacFsm.dac[dacIdx].level = (val << DAC_FRAC_BITS);
    dacFsm.dac[dacIdx].target = val;
}

void dacUpdate(void) {
    DacFsm* fsm = &dacFsm;

    // Update dithering based on frac code
    // 1 frac code step = 1 cycle on PWM, assumes fracCounter incremented every clock cycle
    // When frac code counter below frac code, use code + 1
    // When frac code counter above frac code, use code
    for (int dacIdx = 0; dacIdx < DAC_COUNT; ++dacIdx) {
        Dac* dac = &fsm->dac[dacIdx];
        if (dac->active) {
            const uint16_t level = dac->level;
            const uint16_t levelInt = level >> DAC_FRAC_BITS;
            const uint16_t levelFrac = level & ((1 << DAC_FRAC_BITS) - 1);

            uint32_t fracCounter = dac->fracCounter;
            uint16_t chanLevel = levelInt;
            ++fracCounter;
            if (fracCounter >= (DAC_PWM_WRAP * DAC_FRAC_BITS)) {
                fracCounter = 0;
                chanLevel = levelInt + 1;
            }
            if (fracCounter >= (DAC_PWM_WRAP * levelFrac)) {
                chanLevel = levelInt;
            }
            dac->fracCounter = fracCounter;

            if (dacIdx == 0) {
                pwm_set_chan_level(PWM_SLICE_DAC_OUT_0, PWM_CHAN_DAC_OUT_0, chanLevel);
            }
            else {
                pwm_set_chan_level(PWM_SLICE_DAC_OUT_1, PWM_CHAN_DAC_OUT_1, chanLevel);
            }
        }
    }

    // Update FSM
    switch (fsm->state) {
        case DAC_INIT:
        {
            adc_gpio_init(GPIO_DAC_SENSE_0);
            adc_gpio_init(GPIO_DAC_SENSE_1);

            fsm->state = DAC_MEAS_INIT;
            break;
        }

        case DAC_MEAS_INIT:
        {
            Dac* dac = &fsm->dac[fsm->dacIdx];
            if (dac->enable) {
                dac->active = true;

                // PWMs need to be configured each iteration because it is shared with SOSC
                pwm_config cfg = pwm_get_default_config();
                pwm_config_set_wrap(&cfg, DAC_PWM_WRAP);
                if (fsm->dacIdx == 0) {
                    gpio_set_function(GPIO_DAC_OUT_0, GPIO_FUNC_PWM);
                    adc_select_input(ADC_DAC_SENSE_0);

                    pwm_init(PWM_SLICE_DAC_OUT_0, &cfg, true);
                }
                else {
                    gpio_set_function(GPIO_DAC_OUT_1, GPIO_FUNC_PWM);
                    adc_select_input(ADC_DAC_SENSE_1);

                    pwm_init(PWM_SLICE_DAC_OUT_1, &cfg, true);
                }

                hw_set_bits(&adc_hw->cs, ADC_CS_START_ONCE_BITS);
                fsm->state = DAC_MEAS_WAIT;
            }
            else {
                // Place GPIO pin on high impedance
                if (fsm->dacIdx == 0) {
                    gpio_deinit(GPIO_DAC_OUT_0);
                }
                else {
                    gpio_deinit(GPIO_DAC_OUT_1);
                }

                dac->active = false;
                fsm->state = DAC_MEAS_DONE;
            }

            break;
        }

        case DAC_MEAS_WAIT:
        {
            if (adc_hw->cs & ADC_CS_READY_BITS) {
                uint16_t senseVal = adc_hw->result; // Voltage = result / 2^12

                // Raise duty cycle if output voltage below target
                // Decrease duty cycle if output voltage above target
                Dac* dac = &fsm->dac[fsm->dacIdx];
                uint16_t level = dac->level;
                if (senseVal < dac->target) {
                    if (level < DAC_LEVEL_MAX) {
                        ++level;
                    }
                }
                else {
                    if (level > DAC_LEVEL_MIN) {
                        --level;
                    }
                }

                dac->level = level;
                fsm->state = DAC_MEAS_DONE;
            }
            break;
        }

        case DAC_MEAS_DONE:
        {
            if (fsm->dacIdx == 0) {
                // Measure next DAC
                fsm->dacIdx = 1;
                fsm->state = DAC_MEAS_INIT;
            }
            else {
                fsm->dacIdx = 0;
                fsm->state = DAC_WAIT; // Done measurements, now wait
            }
            break;
        }

        case DAC_WAIT:
        {
            // Wait for output to settle
            ++fsm->waitCounter;
            if (fsm->waitCounter >= DAC_WAIT_CYCLES) {
                fsm->waitCounter = 0;
                fsm->state = DAC_MEAS_INIT;
            }
            break;
        }

        default:
        {
            fsm->state = DAC_INIT;
            break;
        }
    }
}