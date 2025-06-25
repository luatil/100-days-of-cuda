#include <stdio.h>

#define Max(_a, _b) (_a < _b) ? _b : _a

__global__ void PrefixSum(const float *Input, float *Output, const int N)
{
    Output[0] = Input[0];
    for (int I = 1; I < N; I++)
    {
        Output[I] = Input[I] + Output[I-1];
    }
}


int main()
{
    const float HostInput[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float HostExpectedOutput[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float HostOutput[5];

    const int N = 5;

    float *DeviceInput, *DeviceOutput;

    cudaMalloc(&DeviceInput, sizeof(float)*N);
    cudaMalloc(&DeviceOutput, sizeof(float)*N);

    cudaMemcpy(DeviceInput, HostInput, sizeof(float)*N, cudaMemcpyHostToDevice);

    PrefixSum<<<1,1>>>(DeviceInput, DeviceOutput, N);

    cudaMemcpy(HostOutput, DeviceOutput, sizeof(float)*N, cudaMemcpyDeviceToHost);

    float MaxDiff = 0.0f;
    for (int I = 0; I < N; I++)
    {
        float Diff = abs(HostOutput[I] - HostExpectedOutput[I]);
        MaxDiff = Max(Diff, MaxDiff);
        if (Diff > 0.0001f)
        {
            fprintf(stderr, "Diff is too great [%.3f] | Host [%.3f] | Expected [%.3f]", Diff, HostOutput[I], HostExpectedOutput[I]);
            return 1;
        }
    }

    fprintf(stdout, "MaxDiff [%.3f]\n", MaxDiff);
    return 0;
}
    
    
