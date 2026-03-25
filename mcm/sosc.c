#include <hardware/clocks.h>
#include <hardware/gpio.h>
#include <hardware/pwm.h>
#include <hardware/timer.h>

#include "hw_config.h"
#include "sosc.h"
#include "status.h"

enum
{
    SOSC_COUNT          = 2,        // Hardware number of oscillators

    SOSC_POWERUP_TIME_0 = 4000000,  // Time to wait after powering on oscillator before taking measurement [us]
    SOSC_POWERUP_TIME_1 = 1000000,  //   oscillation frequency takes time to settle
    SOSC_MEAS_TIME_0    = 100000,   // Frequency measurement duration [us]
    SOSC_MEAS_TIME_1    = 50000,
    SOSC_CLKDIV_0       = 1,        // Input clock divider, set to avoid overflow 16 bit counter during measurement duration
    SOSC_CLKDIV_1       = 1,

    SOSC_CAL_ITER       = 32,
    SOSC_CAL_ROUND      = 4,        // Rounding when finding mode
    SOSC_CAL_MIN_COUNT  = 1000,     // Minimum frequency to pass calibration: freq / clkdiv * meas_time
                                    // counts = freq / clkdiv * meas_time
                                    // ?? KHz for sosc0, ?? KHz for sosc1

    SOSC_VAR_MAX        = 5,        // Maximum variation expected between measurements

    SOSC_DRIFT_TH       = 4,        // Adjust idleCounts if sign of past measured frequency difference from idleCounts exceeds threshold
    SOSC_VOTE_TH        = 2,        // Consecutive votes required to declare detected
    SOSC_DET_COUNT_TH_0 = 4,        // Count difference from calibrated value for detect vote:
    SOSC_DET_COUNT_TH_1 = 60,       // ?? Hz for sosc0, ?? Hz for sosc1
    SOSC_DET_COUNT_MAX  = 200,      // Ignore large frequency differences from charger pings
};

typedef struct // Data for one oscillator
{
    uint16_t idleCounts;
    uint8_t ready     : 1; // Powered on and ready for measurement
    uint8_t reserved  : 7;
} Sosc;

typedef enum
{
    SOSC_MEAS_INIT = 0,
    SOSC_MEAS_POWERUP,
    SOSC_MEAS_START,
    SOSC_MEAS_WAIT,
} SoscMeasState;

typedef struct
{
    uint64_t startTime;
    SoscMeasState state;
} SoscMeasFsm;

typedef struct
{
    uint16_t countHist[SOSC_CAL_ITER];
    uint8_t iter;
} SoscCalFsm;

typedef struct
{
    int32_t lastDiff0;
    int32_t lastDiff1;
    uint8_t iter;
} SoscDeltaFsm;

typedef struct
{
    int8_t signCounter; // Track drift by counting sign of count differences from idleCounts
    uint8_t detectVote;
} SoscDetFsm;


// Lookups
static uint32_t soscMeasTime[SOSC_COUNT] = {SOSC_MEAS_TIME_0, SOSC_MEAS_TIME_1};
static uint32_t soscPowerupTime[SOSC_COUNT] = {SOSC_POWERUP_TIME_0, SOSC_POWERUP_TIME_1};
static uint32_t soscDetCountTh[SOSC_COUNT] = {SOSC_DET_COUNT_TH_0, SOSC_DET_COUNT_TH_1};
static uint8_t soscPwmSlice[SOSC_COUNT] = {PWM_SLICE_SOSC_0, PWM_SLICE_SOSC_1};
static uint8_t soscEnGpio[SOSC_COUNT] = {GPIO_SOSC_EN_0, GPIO_SOSC_EN_1};
static uint8_t soscOutGpio[SOSC_COUNT] = {GPIO_SOSC_OUT_0, GPIO_SOSC_OUT_1};

static Sosc sosc[SOSC_COUNT];
static SoscMeasFsm soscMeasFsm;
static SoscCalFsm soscCalFsm;
static SoscDeltaFsm soscDeltaFsm;
static SoscDetFsm soscDetFsm;

// Takes frequency measurement on specified oscillator
// Counts stored in pointer when done
bool soscMeas(uint16_t* counts, uint8_t soscIdx) {
    SoscMeasFsm* fsm = &soscMeasFsm;
    bool done = false;
    switch (fsm->state) {
        case SOSC_MEAS_INIT:
        {
            if (!sosc[soscIdx].ready) {
                // Turn on the requested oscillator, and others off
                bool soscEn[SOSC_COUNT] = {};
                soscEn[soscIdx] = true;

                for (uint8_t i = 0; i < SOSC_COUNT; ++i) {
                    gpio_put(soscEnGpio[i], !soscEn[i]); // For PNP off means base at logic high
                    sosc[i].ready = false;
                }

                fsm->startTime = time_us_64();
                fsm->state = SOSC_MEAS_POWERUP;
            }
            else {
                // If oscillator already on, start measurement
                fsm->state = SOSC_MEAS_START;
            }
            break;
        }

        case SOSC_MEAS_POWERUP:
        {
            // Wait for oscillators to power on and settle
            if ((time_us_64() - fsm->startTime) > soscPowerupTime[soscIdx]) {
                sosc[soscIdx].ready = true;
                fsm->state = SOSC_MEAS_START;
            }
            break;
        }

        case SOSC_MEAS_START:
        {
            // Configure PWM and timer for starting measurement
            // PWM needs to be set each measurement since PWM channels are shared with other FSMs
            if (soscIdx == 0) {
                pwm_config cfg = pwm_get_default_config();
                pwm_config_set_clkdiv_mode(&cfg, PWM_DIV_B_RISING);
                pwm_config_set_clkdiv(&cfg, SOSC_CLKDIV_0);

                gpio_set_function(GPIO_SOSC_OUT_0, GPIO_FUNC_PWM);

                pwm_init(PWM_SLICE_SOSC_0, &cfg, true);
            }
            else {
                pwm_config cfg = pwm_get_default_config();
                pwm_config_set_clkdiv_mode(&cfg, PWM_DIV_B_RISING);
                pwm_config_set_clkdiv(&cfg, SOSC_CLKDIV_1);

                gpio_set_function(GPIO_SOSC_OUT_1, GPIO_FUNC_PWM);

                pwm_init(PWM_SLICE_SOSC_1, &cfg, true);
            }

            fsm->startTime = time_us_64();
            fsm->state = SOSC_MEAS_WAIT;
            break;
        }

        case SOSC_MEAS_WAIT:
        {
            // Count number of edge transitions during fixed time interval
            const uint8_t slice = soscPwmSlice[soscIdx];

            // int32_t big enough to hold elapsed time
            uint32_t elapsedTime = (uint32_t)(time_us_64() - fsm->startTime);
            if (elapsedTime >= soscMeasTime[soscIdx]) {
                pwm_set_enabled(slice, false);
                uint16_t rawCounts = pwm_get_counter(slice);

                // The elapsed time may be few ms longer than SOSC_MEAS_TIME
                // Correct counts based on elapsed time, calculate count per us, then scale by extra time
                uint16_t countCorrection = (uint32_t)rawCounts * (elapsedTime - soscMeasTime[soscIdx]) / elapsedTime;
                *counts = rawCounts - countCorrection;

                // Set GPIO to high impedance
                gpio_deinit(soscOutGpio[soscIdx]);

                fsm->state = SOSC_MEAS_INIT;
                done = true;
            }
            break;
        }

        default:
        {
            fsm->state = SOSC_MEAS_INIT;
            break;
        }
    }

    return done;
}

void soscInit(void) {
    // Default to disabled. For PNP means base at logic high
    gpio_init(GPIO_SOSC_EN_0);
    gpio_init(GPIO_SOSC_EN_1);
    gpio_set_dir(GPIO_SOSC_EN_0, GPIO_OUT);
    gpio_set_dir(GPIO_SOSC_EN_1, GPIO_OUT);
}

bool soscCalibrate(uint8_t soscIdx) {
    SoscCalFsm* fsm = &soscCalFsm;
    bool done = false;

    uint16_t counts;
    if (soscMeas(&counts, soscIdx)) {
        if (counts < SOSC_CAL_MIN_COUNT) {
            // Invalid frequency, do the powerup wait again
            sosc[soscIdx].ready = 0;
            fsm->iter = 0;
        }
        else {
            fsm->countHist[fsm->iter] = counts / SOSC_CAL_ROUND;

            ++fsm->iter;
            if (fsm->iter >= SOSC_CAL_ITER) {
                // Find mode in countHist
                // Count occurrences
                uint16_t val[SOSC_CAL_ITER] = {};
                uint8_t valNum[SOSC_CAL_ITER] = {};
                for (uint32_t i = 0; i < SOSC_CAL_ITER; ++i) {
                    for (uint32_t j = 0; j < SOSC_CAL_ITER; ++j) {
                        if (fsm->countHist[i] == val[j]) {
                            valNum[j] += 1;
                            break;
                        }
                        else if (val[j] == 0) {
                            val[j] = fsm->countHist[i];
                            valNum[j] = 1;
                            break;
                        }
                    }
                }
                // Find most occurrences
                uint16_t maxNum = 0;
                uint32_t maxIdx = 0;
                for (uint32_t i = 0; i < SOSC_CAL_ITER; ++i) {
                    if (valNum[i] > maxNum) {
                        maxNum = valNum[i];
                        maxIdx = i;
                    }
                }

                sosc[soscIdx].idleCounts = val[maxIdx] * SOSC_CAL_ROUND; // Rescale back up to original counts

                fsm->iter = 0;
                done = true;
            }
        }
    }
    return done;
}

bool soscDelta(int32_t* countDiff, uint8_t soscIdx) {
    SoscDeltaFsm* fsm = &soscDeltaFsm;

    uint16_t counts;
    if (soscMeas(&counts, soscIdx)) {
        int32_t diff = (int32_t)counts - (int32_t)sosc[soscIdx].idleCounts;

        ++fsm->iter;
        if (fsm->iter >= 3) {
            fsm->iter = 0;

            int32_t variation0 = fsm->lastDiff0 - fsm->lastDiff1;
            if (variation0 < 0) {
                variation0 = -variation0;
            }

            int32_t variation1 = fsm->lastDiff0 - diff;
            if (variation1 < 0) {
                variation1 = -variation1;
            }

            // Check if
            // 1. Edge of charger ping present from variation between first and middle, and middle and last measurements
            //    (1)                                 (2)
            //    Charger ping -------+                       +-------
            //    Measurements <---> <---> <--->     <---> <---> <--->
            // 2. Charger ping within measurement from large counts
            //    Charger ping -----------------
            //    Measurements <---> <---> <--->
            int32_t avgDiff = (diff + fsm->lastDiff0 + fsm->lastDiff1) / 3;
            if (variation0 < SOSC_VAR_MAX && variation1 < SOSC_VAR_MAX && avgDiff < SOSC_DET_COUNT_MAX) {
                *countDiff = avgDiff;
                return true;
            }
        }
        else {
            fsm->lastDiff1 = fsm->lastDiff0;
            fsm->lastDiff0 = diff;
        }
    }
    return false;
}

bool soscDetect(bool* detected, uint8_t soscIdx) {
    SoscDetFsm* fsm = &soscDetFsm;
    bool done = false;

    // Count number of edge transitions during fixed time interval
    int32_t countDiff;
    if (soscDelta(&countDiff, soscIdx)) {
        // Track sign
        // Adjust idle counts if significant drift
        if (countDiff > 0) {
            ++fsm->signCounter;
            if (fsm->signCounter > SOSC_DRIFT_TH) {
                ++sosc[soscIdx].idleCounts;
                fsm->signCounter = 0;
            }
        }
        else if (countDiff < 0) {
            --fsm->signCounter;
            if (fsm->signCounter < -SOSC_DRIFT_TH) {
                --sosc[soscIdx].idleCounts;
                fsm->signCounter = 0;
            }
        }

        // Phone is placed if frequency increased
        if (countDiff > (int32_t)(soscDetCountTh[soscIdx])) {
            ++fsm->detectVote;

            // Detected if sufficient consecutive votes
            if (fsm->detectVote >= SOSC_VOTE_TH) {
                fsm->detectVote = 0;
                *detected = true;
                done = true;
            }
        }
        else {
            fsm->detectVote = 0;

            *detected = false;
            done = true;
        }
    }
    return done;
}