#include "day_015_common.h"

struct random_number_generator
{
    u64 State;
};

static u64 RandomU64(random_number_generator *Rng)
{
    // Simple XorShift
    u64 X = Rng->State;
    X ^= X << 7;
    X ^= X >> 9;
    return Rng->State = X;
}

static random_number_generator Seed(u64 SeedValue)
{
    random_number_generator Result = {};
    Result.State = SeedValue;

    for (u32 I = 0; I < 32; I++)
    {
        RandomU64(&Result);
    }

    return Result;
}

static f32 RandomInRangeF32(random_number_generator *Rng, f32 MinValue, f32 MaxValue)
{
    f32 T = (f32)RandomU64(Rng) / (f32)UINT64_MAX;
    f32 Result = (1.0f - T) * MinValue + T * MaxValue; // Simple Lerp
    return Result;
}

static f32 *MakeRandomF32(u64 SeedValue, u32 N)
{
    f32 *Result = AllocateCPU(f32, N);

    random_number_generator Rng = Seed(SeedValue);

    for (u32 I = 0; I < N; I++)
    {
        Result[I] = RandomInRangeF32(&Rng, -100.f, 100.f);
    }

    return Result;
}

static f32 *MakeSequentialF32(u32 N)
{
    f32 *Result = AllocateCPU(f32, N);

    for (u32 I = 0; I < N; I++)
    {
        Result[I] = (f32)I * 0.01f;
    }

    return Result;
}
