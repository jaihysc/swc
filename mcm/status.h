#ifndef STATUS_H
#define STATUS_H

typedef enum
{
    STATUS_BT_INIT = 0,
    STATUS_BT_ERR,
    STATUS_CAL,
    STATUS_IDLE,
    STATUS_SWEEP_THETA,
    STATUS_SWEEP_RADIUS,
    STATUS_CHARGING,
} StatusCode;

void statusSet(StatusCode status);
void statusUpdate();

#endif // STATUS_H