#include <hardware/clocks.h>
#include <hardware/gpio.h>
#include <hardware/pwm.h>
#include <hardware/timer.h>

#include "hw_config.h"
#include "sosc.h"
#include "status.h"

enum
{
    SOSC_DET_TH         = 7,     // Number of measurements in threshold (votes), of total iterations, to declare detected
    SOSC_MEAS_ITER      = 8,     // Number of measurements (iterations) before making decision
    SOSC_MEAS_TIME      = 250000, // Measurement duration [us]
    SOSC_COUNT_MIN_TH_0 = 5,      // For a measurement: vote threshold: 5 counts  =  20 Hz / clkdiv * meas_time
    SOSC_COUNT_MAX_TH_0 = 50,     // For a measurement: vote threshold: 50 counts = 200 Hz / clkdiv * meas_time
                                  // Significant frequency changes from charger ping ignored
    SOSC_COUNT_MIN_TH_1 = 100,
    SOSC_COUNT_MAX_TH_1 = 1000,
    SOSC_CLKDIV_0       = 1,      // Input clock divider, set to avoid overflow 16 bit counter during measurement duration
    SOSC_CLKDIV_1       = 2,
    SOSC_COUNT          = 2       // Hardware number of oscillators
};

typedef struct // Data for one oscillator
{
    uint64_t startTime;
    uint16_t countsPrev;
    uint8_t detectVote;
    uint8_t iter;
    uint8_t enable    : 1;
    uint8_t active    : 1; // Currently performing measurement
    uint8_t detected  : 1; // Object detected
    uint8_t prevValid : 1; // countsPrev value is valid
    uint8_t reserved  : 4;
} Sosc;

typedef enum
{
    SOSC_INIT = 0,
    SOSC_MEAS_INIT,
    SOSC_MEAS_WAIT,
    SOSC_MEAS_DONE,
} SoscState;

typedef struct
{
    Sosc sosc[SOSC_COUNT];
    SoscState state;
    uint8_t soscIdx;
} SoscFsm;

static SoscFsm soscFsm;

void soscEnable(uint8_t oscIdx, bool enable) {
    soscFsm.sosc[oscIdx].enable = enable;
}

bool soscActive(uint8_t oscIdx) {
    return soscFsm.sosc[oscIdx].active;
}

bool soscDetected(uint8_t oscIdx) {
    return soscFsm.sosc[oscIdx].detected;
}

bool soscCalibrated(uint8_t oscIdx) {
    return soscFsm.sosc[oscIdx].prevValid;
}

void soscUpdate(void) {
    SoscFsm* fsm = &soscFsm;
    switch (fsm->state) {
        case SOSC_INIT:
        {
            // Default to disabled. For PNP means base at logic high
            gpio_init(GPIO_SOSC_EN_0);
            gpio_init(GPIO_SOSC_EN_1);
            gpio_set_dir(GPIO_SOSC_EN_0, GPIO_OUT);
            gpio_set_dir(GPIO_SOSC_EN_1, GPIO_OUT);

            fsm->state = SOSC_MEAS_INIT;
            break;
        }

        case SOSC_MEAS_INIT:
        {
            bool soscEn0 = false;
            bool soscEn1 = false;
            if (fsm->sosc[fsm->soscIdx].enable) {
                fsm->sosc[fsm->soscIdx].active = true;
                // Configure PWM and measure start time
                // This needs to be set each time since PWM channels are shared with other FSMs
                if (fsm->soscIdx == 0) {
                    soscEn0 = true;

                    pwm_config cfg = pwm_get_default_config();
                    pwm_config_set_clkdiv_mode(&cfg, PWM_DIV_B_RISING);
                    pwm_config_set_clkdiv(&cfg, SOSC_CLKDIV_0);

                    gpio_set_function(GPIO_SOSC_OUT_0, GPIO_FUNC_PWM);

                    fsm->sosc[0].startTime = time_us_64();
                    pwm_init(PWM_SLICE_SOSC_0, &cfg, true);
                }
                else {
                    soscEn1 = true;

                    pwm_config cfg = pwm_get_default_config();
                    pwm_config_set_clkdiv_mode(&cfg, PWM_DIV_B_RISING);
                    pwm_config_set_clkdiv(&cfg, SOSC_CLKDIV_1);

                    gpio_set_function(GPIO_SOSC_OUT_1, GPIO_FUNC_PWM);

                    fsm->sosc[1].startTime = time_us_64();
                    pwm_init(PWM_SLICE_SOSC_1, &cfg, true);
                }

                fsm->state = SOSC_MEAS_WAIT;
            }
            else {
                // Do nothing if not enabled
                // Place GPIO pin on high impedance
                if (fsm->soscIdx == 0) {
                    gpio_deinit(GPIO_SOSC_OUT_0);
                }
                else {
                    gpio_deinit(GPIO_SOSC_OUT_1);
                }

                Sosc* sosc = &fsm->sosc[fsm->soscIdx];
                sosc->detected = false;
                sosc->active = false;
                fsm->state = SOSC_MEAS_DONE;
            }

            // For PNP means off means base at logic high
            gpio_put(GPIO_SOSC_EN_0, !soscEn0);
            gpio_put(GPIO_SOSC_EN_1, !soscEn1);
            break;
        }

        case SOSC_MEAS_WAIT:
        {
            // Count number of edge transitions during fixed time interval
            Sosc* sosc = &fsm->sosc[fsm->soscIdx];

            // Load constants and increment index
            uint8_t slice;
            int16_t minTh, maxTh;
            if (fsm->soscIdx == 0) {
                slice = PWM_SLICE_SOSC_0;
                minTh = SOSC_COUNT_MIN_TH_0;
                maxTh = SOSC_COUNT_MAX_TH_0;
            }
            else {
                slice = PWM_SLICE_SOSC_1;
                minTh = SOSC_COUNT_MIN_TH_1;
                maxTh = SOSC_COUNT_MAX_TH_1;
            }

            // int32_t big enough to hold elapsed time
            int32_t elapsedTime = (int32_t)(time_us_64() - sosc->startTime);
            if (elapsedTime >= SOSC_MEAS_TIME) {
                pwm_set_enabled(slice, false);
                uint16_t counts = pwm_get_counter(slice);

                // The elapsed time may be few ms longer than SOSC_MEAS_TIME
                // Correct counts based on elapsed time, calculate count per us, then scale by extra time
                uint16_t countCorrection = (uint32_t)counts * (elapsedTime - SOSC_MEAS_TIME) / elapsedTime;
                counts = counts - countCorrection;

                if (sosc->prevValid) {
                    int32_t countDiff = counts - sosc->countsPrev;
                    bool diffNegative = countDiff < 0;

                    int32_t absCountDiff = countDiff;
                    if (diffNegative) {
                        absCountDiff = -absCountDiff;
                    }

                    // Phone is placed if frequency increased
                    if (!diffNegative && absCountDiff > minTh && absCountDiff < maxTh) {
                        ++sosc->detectVote;
                    }

                    ++sosc->iter;
                    if (sosc->iter >= SOSC_MEAS_ITER) {
                        bool detected = false;
                        if (sosc->detectVote >= SOSC_DET_TH) {
                            detected = true;
                        }
                        else if (sosc->detectVote == 0) {
                            // Update counts when certain is idle

                            // Don't update if difference is small
                            // This fixes issue where slowly moving phone towards table not detected as the countsPrev slowly inc/decrements
                            if (absCountDiff > minTh) {
                                sosc->countsPrev = counts;
                            }
                        }
                        sosc->detected = detected;

                        // Reset
                        sosc->detectVote = 0;
                        sosc->iter = 0;
                    }
                }
                else {
                    // At startup, calculate countsPrev
                    sosc->countsPrev = counts;
                    sosc->prevValid = true;
                }

                fsm->state = SOSC_MEAS_DONE;
            }
            break;
        }

        case SOSC_MEAS_DONE:
        {
            // Next oscillator
            if (fsm->soscIdx == 0) {
                fsm->soscIdx = 1;
            }
            else {
                fsm->soscIdx = 0;
            }
            fsm->state = SOSC_MEAS_INIT;
            break;
        }

        default:
        {
            fsm->state = SOSC_INIT;
            break;
        }
    }
}