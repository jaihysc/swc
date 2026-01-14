#include <hardware/clocks.h>
#include <hardware/gpio.h>
#include <hardware/pwm.h>
#include <hardware/timer.h>

#include "hw_config.h"
#include "sosc.h"
#include "status.h"

enum
{
    SOSC_DET_TH         = 10,     // Number of measurements in threshold (votes), of total iterations, to declare detected
    SOSC_MEAS_ITER      = 16,     // Number of measurements (iterations) before making decision
    SOSC_MEAS_TIME      = 250000, // Measurement duration [us]
    SOSC_COUNT_MIN_TH_0 = 10,     // For a measurement: vote threshold: 10 counts  =  40 Hz / clkdiv * meas_time
    SOSC_COUNT_MAX_TH_0 = 100,    // For a measurement: vote threshold: 100 counts = 400 Hz / clkdiv * meas_time
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
    bool detected;
} SoscData;

typedef enum
{
    SOSC_INIT = 0,
    SOSC_MEAS,
} SoscState;

typedef struct
{
    SoscData sosc[SOSC_COUNT];
    SoscState state;
    bool hasCountsPrev;
    uint8_t soscIdx;
} Sosc;

static Sosc sosc;

void soscUpdate(void) {
    Sosc* fsm = &sosc;

    // Pulse LED if detected
    if (fsm->sosc[1].detected) {
        statusSet(STATUS_SOSC_1_DET);
    }
    else if (fsm->sosc[0].detected) {
        statusSet(STATUS_SOSC_0_DET);
    }
    else {
        statusSet(STATUS_IDLE);
    }

    switch (fsm->state) {
        case SOSC_INIT:
        {
            pwm_config cfg = pwm_get_default_config();
            pwm_config_set_clkdiv_mode(&cfg, PWM_DIV_B_RISING);
            pwm_config_set_clkdiv(&cfg, SOSC_CLKDIV_0);
            pwm_init(PWM_SLICE_SOSC_0, &cfg, false);

            pwm_config_set_clkdiv(&cfg, SOSC_CLKDIV_1);
            pwm_init(PWM_SLICE_SOSC_1, &cfg, false);

            gpio_set_function(GPIO_SOSC_0, GPIO_FUNC_PWM);
            gpio_set_function(GPIO_SOSC_1, GPIO_FUNC_PWM);

            fsm->sosc[0].startTime = time_us_64();
            pwm_set_enabled(PWM_SLICE_SOSC_0, true);

            fsm->sosc[1].startTime = time_us_64();
            pwm_set_enabled(PWM_SLICE_SOSC_1, true);

            fsm->state = SOSC_MEAS;
            break;
        }

        case SOSC_MEAS:
        {
            // Count number of edge transitions during fixed time interval
            SoscData* sosc = &fsm->sosc[fsm->soscIdx];
            uint8_t slice;
            int16_t minTh, maxTh;
            if (fsm->soscIdx == 0) {
                slice = PWM_SLICE_SOSC_0;
                minTh = SOSC_COUNT_MIN_TH_0;
                maxTh = SOSC_COUNT_MAX_TH_0;
                fsm->soscIdx = 1;
            }
            else {
                slice = PWM_SLICE_SOSC_1;
                minTh = SOSC_COUNT_MIN_TH_1;
                maxTh = SOSC_COUNT_MAX_TH_1;
                fsm->soscIdx = 0;
            }

            // int32_t big enough to hold elapsed time
            int32_t elapsedTime = (int32_t)(time_us_64() - sosc->startTime);
            if (elapsedTime >= SOSC_MEAS_TIME) {
                pwm_set_enabled(slice, false);
                uint16_t counts = pwm_get_counter(slice);
                pwm_set_counter(slice, 0);

                // The elapsed time may be few ms longer than SOSC_MEAS_TIME
                // Correct counts based on elapsed time, calculate count per us, then scale by extra time
                uint16_t countCorrection = (uint32_t)counts * (elapsedTime - SOSC_MEAS_TIME) / elapsedTime;
                counts = counts - countCorrection;

                if (fsm->hasCountsPrev) {
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

                            // Update counts to avoid re-detection
                            sosc->countsPrev = counts;
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
                    fsm->hasCountsPrev = true;
                }

                // Measure again
                sosc->startTime = time_us_64();
                pwm_set_enabled(slice, true);
            }
            break;
        }

        default:
        {
            fsm->state = SOSC_INIT;
            break;
        }
    }
}