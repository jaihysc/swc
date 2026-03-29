#ifndef SOSC_H
#define SOSC_H

// Configures hardware
void soscInit(void);

// Powers up specified oscillator
void soscPowerup(uint8_t soscIdx);

// Takes frequency measurement on specified oscillator
// Counts stored in pointer when done
bool soscMeas(uint16_t* counts, uint8_t soscIdx);

// Determines oscillator frequency when no phones present
bool soscCalibrate(uint8_t soscIdx);

// Determines oscillator average counts, rejects charger pings
bool soscCounts(int32_t* avgCounts, uint8_t soscIdx);

// Measures sense oscillator frequency to detect objects
bool soscDetect(bool* detected, uint8_t soscIdx);

#endif // SOSC_H