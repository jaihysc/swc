#ifndef SOSC_H
#define SOSC_H

// Enable/disable oscillator
// To disable, wait until soscActive returns false
void soscEnable(uint8_t oscIdx, bool enable);

// If oscillator is active (enabled)
bool soscActive(uint8_t oscIdx);

// If phone was detected
bool soscDetected(uint8_t oscIdx);

// If oscillator base frequency has been determined
// Objects are detected if frequency differs from base significantly
bool soscCalibrated(uint8_t oscIdx);

// Measures sense oscillators to detect objects
void soscUpdate(void);

#endif // SOSC_H