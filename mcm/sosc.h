#ifndef SOSC_H
#define SOSC_H

// Configures hardware
void soscInit(void);

bool soscMeas(uint16_t* counts, uint8_t soscIdx);

// Determines oscillator frequency when no phones present
bool soscCalibrate(uint8_t soscIdx);

// Difference in counts from calibrated value, rejects charger pings
bool soscDelta(int32_t* countDiff, uint8_t soscIdx);

// Measures sense oscillator frequency to detect objects
bool soscDetect(bool* detected, uint8_t soscIdx);

#endif // SOSC_H