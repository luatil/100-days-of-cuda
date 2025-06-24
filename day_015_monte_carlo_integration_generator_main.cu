/*
 * ./exe [seed] [number of values]
 *
 * 0.0f 3.0f
 * 8
 * 0.0625 0.25 0.5625 1.0 1.5625 2.25 3.0625 4.0
 *
 */
#include <stdint.h>
#include <stdio.h>

typedef float f32;
typedef unsigned int u32;

#define Min(_A, _B) ((_A < _B) ? _A : _B)
#define Max(_A, _B) ((_A > _B) ? _A : _B)

static u32 ClampU32(u32 MinValue, u32 Value, u32 MaxValue)
{
    u32 Result = Max(MinValue, Min(MaxValue, Value));
    return Result;
}

struct xorshift32_state
{
    u32 State;
};

/* The state must be initialized to non-zero */
u32 XorShift32(xorshift32_state *State)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    u32 x = State->State;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return State->State = x;
}

static f32 RandomZeroToOne(xorshift32_state *State)
{
    u32 NextState = XorShift32(State);
    f32 NormalizedState = (f32)NextState / (f32)UINT32_MAX;
    return NormalizedState;
}

static f32 RandomInRange(xorshift32_state *State, f32 Min, f32 Max)
{
    f32 ZeroToOne = RandomZeroToOne(State);
    f32 Result = (1.0f - ZeroToOne) * Min + ZeroToOne * Max;
    return Result;
}

int main(int Argc, char **Argv)
{
    if (Argc == 3)
    {
        u32 Seed;
        u32 NumberOfValues;

        sscanf(Argv[1], "%u", &Seed);
        sscanf(Argv[2], "%u", &NumberOfValues);

        NumberOfValues = ClampU32(0, NumberOfValues, 100000000);

        xorshift32_state State = {};
        State.State = Seed;

        for (u32 I = 0; I < 1024; I++)
        {
            XorShift32(&State);
        }

        f32 A = RandomInRange(&State, -1000.0f, 1000.0f);
        f32 B = RandomInRange(&State, -1000.0f, 1000.0f);

        fprintf(stdout, "%f %f\n", Min(A, B), Max(A, B));
        fprintf(stdout, "%d\n", NumberOfValues);

        for (u32 I = 0; I < NumberOfValues; I++)
        {
            f32 GeneratedValue = RandomInRange(&State, -10000.0, 10000.0);
            fprintf(stdout, "%f ", GeneratedValue);
        }

        fprintf(stdout, "\n");
    }
    else
    {
        fprintf(stderr, "Usage: %s [seed] [number of values]\n", Argv[0]);
    }
}
