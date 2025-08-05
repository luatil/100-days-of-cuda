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

#include "day_001_macros.h"
#include "day_015_monte_carlo_integration_kernels.cu"

#define ARRAY_COUNT(_Array) sizeof(_Array) / sizeof(_Array[0])

typedef unsigned int u32;
typedef float f32;

struct test_function
{
    char const *Name;
    monte_carlo_integration_function *Func;
};
test_function TestFunctions[] = {
    {"cpu", Launch_MonteCarloIntegration_CPU},
    {"gpu_naive", Launch_MonteCarloIntegration_Naive},
};

static void PrintUsage(const char *ProgramName)
{
    fprintf(stderr, "Usage: %s [", ProgramName);
    for (u32 I = 0; I < ARRAY_COUNT(TestFunctions); I++)
    {
        const char *Separator = (I == ARRAY_COUNT(TestFunctions) - 1) ? "]" : " | ";
        fprintf(stderr, "%s%s", TestFunctions[I].Name, Separator);
    }
    fprintf(stderr, "\n");
}

int main(int Argc, char **Argv)
{
    if (Argc == 2)
    {
        f32 A, B;
        scanf("%f %f", &A, &B);
        u32 NumberOfValues;
        scanf("%u", &NumberOfValues);

        f32 *Array = AllocateCPU(f32, NumberOfValues);

        for (u32 I = 0; I < NumberOfValues; I++)
        {
            scanf("%f", Array + I);
        }

        for (u32 I = 0; I < ARRAY_COUNT(TestFunctions); I++)
        {
            if (strcmp(TestFunctions[I].Name, Argv[1]) == 0)
            {
                test_function *Func = TestFunctions + I;
                f32 Result = Func->Func(Array, A, B, NumberOfValues);
                fprintf(stdout, "%f\n", Result);
            }
        }
    }
    else
    {
        PrintUsage(Argv[0]);
    }
}
