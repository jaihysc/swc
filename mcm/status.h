#ifndef STATUS_H
#define STATUS_H

typedef enum
{
    STATUS_BT_INIT = 0,
    STATUS_BT_ERR,
    STATUS_CAL_0,
    STATUS_CAL_1,
    STATUS_IDLE,
    STATUS_SWEEP_THETA,
    STATUS_SWEEP_RADIUS,
    STATUS_CHARGING,
    STATUS_RESET_RADIUS,
    STATUS_RESET_THETA,
} StatusCode;

void statusSet(StatusCode status);
void statusUpdate();

#endif // STATUS_H