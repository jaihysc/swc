#ifndef HW_CONFIG_H
#define HW_CONFIG_H

enum
{
    GPIO_STATUS_LED     = 0,
    GPIO_SOSC_0         = 1,
    GPIO_SOSC_1         = 3,
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
};

#endif // HW_CONFIG_H