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

static u8 RandomColor(u8 Min, u8 Max)
{
    return (u8)Min + rand() % (Max - Min + 1);
}

#endif
