#ifndef DAC_H
#define DAC_H

// Enable/disable DAC
// To disable, wait until dacActive returns false
void dacEnable(uint8_t dacIdx, bool enable);

// Query if DAC is active (enabled)
bool dacActive(uint8_t dacIdx);

// Sets voltage value of DAC
// Voltage = val/iovdd * 2^12
void dacSet(uint8_t dacIdx, uint16_t val);

// Update function for DACs, call in main loop
void dacUpdate(void);

#endif // DAC_H