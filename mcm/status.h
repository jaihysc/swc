#ifndef STATUS_H
#define STATUS_H

typedef enum
{
    STATUS_BT_INIT = 0,
    STATUS_BT_ERR,
    STATUS_IDLE,
    STATUS_TRANSIT,
    STATUS_SOSC_0_DET,
    STATUS_SOSC_1_DET
} StatusCode;

void statusSet(StatusCode status);
void statusUpdate(); // Call priodically in main loop

#endif // STATUS_H