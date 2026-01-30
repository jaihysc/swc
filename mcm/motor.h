#ifndef MOTOR_H
#define MOTOR_H

// If motor is at target and ready for next move command
bool motorReady(uint8_t motIdx);

// Request motor move for set steps, sign of steps is direction
void motorMove(uint8_t motIdx, int32_t steps);

void motorUpdate(void);

#endif // MOTOR_H