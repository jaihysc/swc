#ifndef HW_CONFIG_H
#define HW_CONFIG_H

enum
{
#ifdef PICO2_W
    GPIO_STATUS_LED     = 0,
    GPIO_SOSC_OUT_0     = 1,
    GPIO_SOSC_EN_0      = 2,
    GPIO_SOSC_OUT_1     = 3,
    GPIO_SOSC_EN_1      = 4,
    GPIO_DAC_OUT_0      = 21,
    GPIO_DAC_SENSE_0    = 26, // ADC0
    GPIO_DAC_OUT_1      = 22,
    GPIO_DAC_SENSE_1    = 27, // ADC1

    PWM_SLICE_SOSC_0    = 0,
    PWM_SLICE_SOSC_1    = 1,
    PWM_SLICE_DAC_OUT_0 = 2,
    PWM_SLICE_DAC_OUT_1 = 3,
    PWM_CHAN_DAC_OUT_0  = 1,
    PWM_CHAN_DAC_OUT_1  = 0,

    // GPIO26 = ADC0
    ADC_DAC_SENSE_0     = GPIO_DAC_SENSE_0 - 26,
    ADC_DAC_SENSE_1     = GPIO_DAC_SENSE_1 - 26,
#else // Pico 1 or RP2040
    GPIO_STATUS_LED     = 25,
    GPIO_SOSC_OUT_0     = 3, // Board rev2 incorrectly has this on GPIO 4
    GPIO_SOSC_EN_0      = 5,
    GPIO_SOSC_OUT_1     = 1,
    GPIO_SOSC_EN_1      = 0,
    GPIO_DAC_OUT_0      = 16,
    GPIO_DAC_SENSE_0    = 28, // ADC2
    GPIO_DAC_OUT_1      = 15,
    GPIO_DAC_SENSE_1    = 27, // ADC1

    PWM_SLICE_SOSC_0    = 1,
    PWM_SLICE_SOSC_1    = 0, // Slice 0 shared between sosc1 and dac0
    PWM_SLICE_DAC_OUT_0 = 0,
    PWM_SLICE_DAC_OUT_1 = 7,
    PWM_CHAN_DAC_OUT_0  = 0,
    PWM_CHAN_DAC_OUT_1  = 1,

    // GPIO26 = ADC0
    ADC_DAC_SENSE_0     = GPIO_DAC_SENSE_0 - 26,
    ADC_DAC_SENSE_1     = GPIO_DAC_SENSE_1 - 26,
#endif
};

#endif // HW_CONFIG_H