#ifndef RANDOM_H
#define RANDOM_H

#include "day_051_types.h"

static f32 RandomFloat(f32 Min, f32 Max)
{
    return Min + (f32)rand() / RAND_MAX * (Max - Min);
}

static u8 RandomColor()
{
    return (u8)(100 + rand() % 156);
}

#endif
