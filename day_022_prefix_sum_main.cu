#include <stdio.h>
#define LEET_GPU_NO_IMPORT

#define Max(_a, _b) _a > _b ? _a : _b

// #include "day_022_prefix_sum_02.cu"
#include "day_022_prefix_sum_02.cu"

int main()
{
    float Input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float Expected[] = {1.0f, 3.0f, 6.0f, 10.0f};
    float Output[sizeof(Input) / sizeof(Input[0])] = {};

    float *D_Output, *D_Input;
    cudaMalloc(&D_Input, sizeof(Input));
    cudaMalloc(&D_Output, sizeof(Input));

    cudaMemcpy(D_Input, Input, sizeof(Input), cudaMemcpyHostToDevice);

    solve(D_Input, D_Output, sizeof(Input) / sizeof(Input[0]));

    cudaMemcpy(Output, D_Output, sizeof(Input), cudaMemcpyDeviceToHost);

    float MaxDiff = 0.0f;
    for (unsigned int I = 0; I < (sizeof(Input) / sizeof(Input[0])); I++)
    {
        float Diff = abs(Output[I] - Expected[I]);
        MaxDiff = Max(Diff, MaxDiff);
        if (Diff > 0.00001f)
        {
            fprintf(stderr, "Expected [%.5f] does not match with Actual Output: [%.5f]", Expected[I], Output[I]);
            return 1;
        }
    }
    fprintf(stdout, "Maximun Difference [%.5f]", MaxDiff);
    return 0;
}
