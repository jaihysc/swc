#ifndef DAC_H
#define DAC_H

// Sets voltage value of DAC
// Voltage = val/iovdd * 2^12
void dacSet(uint8_t dacIdx, uint16_t val);

// Update function for DACs, call in main loop
void dacUpdate(void);

#endif // DAC_H