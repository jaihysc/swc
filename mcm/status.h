#ifndef STATUS_H
#define STATUS_H

typedef enum
{
    STATUS_BT_INIT = 0,
    STATUS_BT_ERR,
    STATUS_OK,
    STATUS_TRANSIT,
} StatusCode;

void statusSet(StatusCode status);
void statusUpdate(); // Call priodically in main loop

#endif // STATUS_H