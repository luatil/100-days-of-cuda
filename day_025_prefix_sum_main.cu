#include <stdio.h>

#define MAX(_a, _b) (_a < _b) ? _b : _a

#define BLOCK_DIM 256

__global__ void CalculatePartialPrefixSums(const float *Input, float *Output, float *PartialPrefixSums, const int N)
{
    __shared__ float Shared[BLOCK_DIM];

    const int TID = blockDim.x * blockIdx.x + threadIdx.x;
    const int TX = threadIdx.x;

    Shared[TX] = TID < N ? Input[TID] : 0.0f;
    __syncthreads();

    for (int Stride = 1; Stride <= blockDim.x / 2; Stride *= 2)
    {
        float Temp = 0.0f;
        if (TX >= Stride)
        {
            Temp = Shared[TX] + Shared[TX - Stride];
        }
        __syncthreads();
        if (TX >= Stride)
        {
            Shared[TX] = Temp;
        }
        __syncthreads();
    }

    if (TID < N)
    {
        Output[TID] = Shared[TX];
    }

    if (TX == BLOCK_DIM - 1)
    {
        // PartialSums[blockIdx.x] = Shared[Tx];
    }
}

// Input and Output are device pointers
static void PrefixSum(const float *Input, float *Output, const int N)
{
    const int GRID_DIM = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    float *PartialPrefixSums = 0;
    CalculatePartialPrefixSums<<<GRID_DIM, BLOCK_DIM>>>(Input, Output, PartialPrefixSums, N);
    // SimplePrefixSum<<<1,1>>>(PartialPrefixSums, GridDim);
    // ExpandPartialSums<<<GridDim, BLOCK_DIM>>>(Output, PartialPrefixSums, N);
}

int main()
{
    const float HOST_INPUT[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float HOST_EXPECTED_OUTPUT[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float HostOutput[5];

    const int N = 5;

    float *DeviceInput, *DeviceOutput;

    cudaMalloc(&DeviceInput, sizeof(float) * N);
    cudaMalloc(&DeviceOutput, sizeof(float) * N);

    cudaMemcpy(DeviceInput, HOST_INPUT, sizeof(float) * N, cudaMemcpyHostToDevice);

    PrefixSum(DeviceInput, DeviceOutput, N);

    cudaMemcpy(HostOutput, DeviceOutput, sizeof(float) * N, cudaMemcpyDeviceToHost);

    float MaxDiff = 0.0f;
    for (int I = 0; I < N; I++)
    {
        float Diff = abs(HostOutput[I] - HOST_EXPECTED_OUTPUT[I]);
        MaxDiff = MAX(Diff, MaxDiff);
        if (Diff > 0.0001f)
        {
            fprintf(stderr, "Diff is too great [%.3f] | Host [%.3f] | Expected [%.3f]", Diff, HostOutput[I],
                    HOST_EXPECTED_OUTPUT[I]);
            return 1;
        }
    }

    fprintf(stdout, "MaxDiff [%.3f]\n", MaxDiff);
    return 0;
}
