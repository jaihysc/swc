#include <hardware/clocks.h>
#include <hardware/gpio.h>
#include <hardware/pwm.h>
#include <hardware/timer.h>

#include "hw_config.h"
#include "sosc.h"
#include "status.h"

enum
{
    SOSC_DET_TH       = 10,     // Number of measurements in threshold (votes), of total iterations, to declare detected
    SOSC_MEAS_ITER    = 16,     // Number of measurements (iterations) before making decision
    SOSC_MEAS_TIME    = 250000, // Measurement duration [us]
    SOSC_COUNT_MIN_TH = 10,     // For a measurement: vote threshold: 10 counts  =  40 Hz / clkdiv * meas_time
    SOSC_COUNT_MAX_TH = 100,    // For a measurement: vote threshold: 100 counts = 400 Hz / clkdiv * meas_time
                                // Significant frequency changes from charger ping ignored
    SOSC_CLKDIV       = 1,      // Input clock divider, set to avoid overflow 16 bit counter during measurement duration
};

typedef enum
{
    SOSC_INIT = 0,
    SOSC_MEAS,
} SoscState;

typedef struct
{
    uint64_t startTime;
    uint16_t countsPrev;
    SoscState state;
    bool hasCountsPrev;
    uint8_t detectVote;
    uint8_t iter;
} Sosc;

static Sosc sosc;


void soscUpdate(void) {
    Sosc* fsm = &sosc;
    switch (fsm->state) {
        case SOSC_INIT:
        {
            fsm->startTime = time_us_64();

            pwm_config cfg = pwm_get_default_config();
            pwm_config_set_clkdiv_mode(&cfg, PWM_DIV_B_RISING);
            pwm_config_set_clkdiv(&cfg, SOSC_CLKDIV);
            pwm_init(PWM_SLICE_SOSC_0, &cfg, false);

            gpio_set_function(GPIO_SOSC_0, GPIO_FUNC_PWM);

            pwm_set_enabled(PWM_SLICE_SOSC_0, true);

            fsm->state = SOSC_MEAS;
            break;
        }

        case SOSC_MEAS:
        {
            // Count number of edge transitions during fixed time interval

            // int32_t big enough to hold elapsed time
            int32_t elapsedTime = (int32_t)(time_us_64() - fsm->startTime);
            if (elapsedTime >= SOSC_MEAS_TIME) {
                pwm_set_enabled(PWM_SLICE_SOSC_0, false);
                int16_t counts = pwm_get_counter(PWM_SLICE_SOSC_0);
                pwm_set_counter(PWM_SLICE_SOSC_0, 0);

                // The elapsed time may be few ms longer than SOSC_MEAS_TIME
                // Correct counts based on elapsed time, calculate count per us, then scale by extra time
                int16_t countCorrection = (int32_t)counts * (elapsedTime - SOSC_MEAS_TIME) / elapsedTime;
                counts = counts - countCorrection;

                if (fsm->hasCountsPrev) {
                    int16_t absCountDiff = counts - fsm->countsPrev;
                    if (absCountDiff < 0) {
                        absCountDiff = -absCountDiff;
                    }

                    if (absCountDiff > SOSC_COUNT_MIN_TH && absCountDiff < SOSC_COUNT_MAX_TH) {
                        ++fsm->detectVote;
                    }

                    ++fsm->iter;
                    if (fsm->iter >= SOSC_MEAS_ITER) {
                        bool detected = false;
                        if (fsm->detectVote >= SOSC_DET_TH) {
                            detected = true;
                            // Update counts to avoid re-detection
                            fsm->countsPrev = counts;
                        }
                        else if (fsm->detectVote == 0) {
                            // Update counts when certain is idle

                            // Don't update if difference is small
                            // This fixes issue where slowly moving phone towards table not detected as the countsPrev slowly inc/decrements
                            if (absCountDiff > SOSC_COUNT_MIN_TH) {
                                fsm->countsPrev = counts;
                            }
                        }

                        // Pulse LED if detected
                        if (detected) {
                            statusSet(STATUS_TRANSIT);
                        }
                        else {
                            statusSet(STATUS_IDLE);
                        }

                        // Reset
                        fsm->detectVote = 0;
                        fsm->iter = 0;
                    }
                }
                else {
                    // At startup, calculate countsPrev
                    fsm->countsPrev = counts;
                    fsm->hasCountsPrev = true;
                }

                // Measure again
                fsm->startTime = time_us_64();
                pwm_set_enabled(PWM_SLICE_SOSC_0, true);
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