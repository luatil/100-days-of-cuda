#include <stdio.h>

#define Max(_a, _b) (_a < _b) ? _b : _a

#define BLOCK_DIM 256

__global__ void CalculatePartialPrefixSums(const float *Input, float *Output, float *PartialPrefixSums, const int N)
{
    __shared__ float Shared[BLOCK_DIM];

    const int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int Tx = threadIdx.x;

    Shared[Tx] = Tid < N ? Input[Tid] : 0.0f;
    __syncthreads();

    for(int Stride = 1; Stride <= blockDim.x / 2; Stride *= 2)
    {
        float Temp = 0.0f;
        if (Tx >= Stride)
        {
            Temp = Shared[Tx] + Shared[Tx - Stride];
        }
        __syncthreads();
        if (Tx >= Stride)
        {
            Shared[Tx] = Temp;
        }
        __syncthreads();
    }

    if (Tid < N)
    {
        Output[Tid] = Shared[Tx];
    }

    if (Tx == BLOCK_DIM - 1)
    {
        // PartialSums[blockIdx.x] = Shared[Tx];
    }
}

// Input and Output are device pointers
static void PrefixSum(const float *Input, float *Output, const int N)
{
    const int GridDim = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    float *PartialPrefixSums = 0;
    CalculatePartialPrefixSums<<<GridDim, BLOCK_DIM>>>(Input, Output, PartialPrefixSums, N);
    // SimplePrefixSum<<<1,1>>>(PartialPrefixSums, GridDim);
    // ExpandPartialSums<<<GridDim, BLOCK_DIM>>>(Output, PartialPrefixSums, N);
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

    PrefixSum(DeviceInput, DeviceOutput, N);

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
    
    
