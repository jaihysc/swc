#ifndef SOSC_H
#define SOSC_H

// Configures hardware
void soscInit(void);

// Determines oscillator frequency when no phones present
bool soscCalibrate(uint8_t soscIdx);

// Measures sense oscillator frequency to detect objects
bool soscDetect(bool* detected, uint8_t soscIdx);

#endif // SOSC_H