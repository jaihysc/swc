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

    SOSC_POWERUP_TIME_0 = 8000000,  // Time to wait after powering on oscillator before taking measurement [us]
    SOSC_POWERUP_TIME_1 = 2000000,  //   oscillation frequency takes time to settle
    SOSC_MEAS_TIME_0    = 500000,   // Frequency measurement duration [us]
    SOSC_MEAS_TIME_1    = 250000,
    SOSC_CLKDIV_0       = 1,        // Input clock divider, set to avoid overflow 16 bit counter during measurement duration
    SOSC_CLKDIV_1       = 2,

    SOSC_CAL_ITER       = 64,
    SOSC_CAL_MIN_COUNT  = 5000,     // Minimum frequency to pass calibration: freq / clkdiv * meas_time
                                    // counts = freq / clkdiv * meas_time
                                    // 20 KHz for sosc0, 40 KHz for sosc1

    SOSC_VOTE_TH        = 3,        // Consecutive votes required to declare detected
    SOSC_DET_ITER       = 4,        // Number of measurements (iterations) before making vote detect/not detected
    SOSC_DET_COUNT_TH_0 = 1,        // Count difference from calibrated value for detect vote:
    SOSC_DET_COUNT_TH_1 = 100,      // 12 Hz for sosc0, 800 Hz for sosc1
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
    uint32_t countsAccum;
    uint8_t iter;
} SoscCalFsm;

typedef struct
{
    uint32_t countsAccum;
    uint8_t detectVote;
    uint8_t detIter;
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
static SoscDetFsm soscDetFsm;

// Takes frequency measurement on specified oscillator
// Counts stored in pointer when done
static bool measFreq(uint16_t* counts, uint8_t soscIdx) {
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
            int32_t elapsedTime = (int32_t)(time_us_64() - fsm->startTime);
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
    if (measFreq(&counts, soscIdx)) {
        if (counts < SOSC_CAL_MIN_COUNT) {
            // Invalid frequency, do the powerup wait again
            sosc[soscIdx].ready = 0;
            fsm->countsAccum = 0;
            fsm->iter = 0;
        }
        else {
            fsm->countsAccum += counts;

            ++fsm->iter;
            if (fsm->iter >= SOSC_CAL_ITER) {
                sosc[soscIdx].idleCounts = fsm->countsAccum / SOSC_CAL_ITER;

                fsm->countsAccum = 0;
                fsm->iter = 0;
                done = true;
            }
        }
    }
    return done;
}

bool soscDetect(bool* detected, uint8_t soscIdx) {
    SoscDetFsm* fsm = &soscDetFsm;
    bool done = false;

    // Count number of edge transitions during fixed time interval
    uint16_t counts;
    if (measFreq(&counts, soscIdx)) {
        fsm->countsAccum += counts;

        ++fsm->detIter;
        if (fsm->detIter >= SOSC_DET_ITER) {
            int32_t countDiff = (int32_t)(fsm->countsAccum / SOSC_DET_ITER) - (int32_t)sosc[soscIdx].idleCounts;

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

            // Reset
            fsm->countsAccum = 0;
            fsm->detIter = 0;
        }
    }
    return done;
}