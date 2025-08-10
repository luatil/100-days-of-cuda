#include <stdio.h>
#define LEET_GPU_NO_IMPORT

#define MAX(_a, _b) _a > _b ? _a : _b

// #include "day_022_prefix_sum_02.cu"
// #include "day_022_prefix_sum_02.cu"
#include "day_022_prefix_sum_03.cu"

int main()
{
    float Input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float Expected[] = {1.0f, 3.0f, 6.0f, 10.0f};
    float Output[sizeof(Input) / sizeof(Input[0])] = {};

    float *DOutput, *DInput;
    cudaMalloc(&DInput, sizeof(Input));
    cudaMalloc(&DOutput, sizeof(Input));

    cudaMemcpy(DInput, Input, sizeof(Input), cudaMemcpyHostToDevice);

    // solve(DInput, DOutput, sizeof(Input) / sizeof(Input[0]));

    cudaMemcpy(Output, DOutput, sizeof(Input), cudaMemcpyDeviceToHost);

    float MaxDiff = 0.0f;
    for (unsigned int I = 0; I < (sizeof(Input) / sizeof(Input[0])); I++)
    {
        float Diff = abs(Output[I] - Expected[I]);
        MaxDiff = MAX(Diff, MaxDiff);
        if (Diff > 0.00001f)
        {
            fprintf(stderr, "Expected [%.5f] does not match with Actual Output: [%.5f]", Expected[I], Output[I]);
            return 1;
        }
    }
    fprintf(stdout, "Maximun Difference [%.5f]", MaxDiff);
    return 0;
}
