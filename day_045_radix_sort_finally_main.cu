/*
 * NAME
 * 	rsort - sorts data using the gpu with radix sort
 *
 * SYNOPSIS
 * 	rsort
 *
 * DESCRIPTION
 * 	Sorts unsigned integer input lines using radix sort to stdout.
 *
 * USAGE
 * 	Sort numbers from 1 to 1000 after shuffling
 * 	- seq 1 1000 | shuf | rsort
 *
 */
#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_VALUE 100000000
#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

static unsigned int Lines[MAX_VALUE];

__device__ __host__ void Swap(unsigned int **A, unsigned int **B)
{
    unsigned int *Temp = *A;
    *A = *B;
    *B = Temp;
}

template <int BLOCK_DIM>
__global__ void ExclusiveSumKernel(const unsigned int *XS, unsigned int *Output, unsigned int *BlockCounter,
                                   unsigned int *Flags, unsigned int *ScanValue, int Bit, int N)
{
    // Dynamically select BlockId based on scheduling order
    __shared__ unsigned int BidS;
    if (threadIdx.x == 0)
    {
        BidS = atomicAdd(BlockCounter, 1);
    }
    __syncthreads();
    unsigned int Bid = BidS;

    __shared__ unsigned int Shared[BLOCK_DIM];

    // For now no thread coarsening
    int Tid = blockDim.x * Bid + threadIdx.x;
    // int Tid = blockDim.x * blockIdx.x + threadIdx.x; // ← Correct indexing
    int Tx = threadIdx.x;

    Shared[Tx] = Tid < N ? ((XS[Tid] >> Bit) & 1) : 0;
    __syncthreads();

    // Perform inclusive scan within block
    for (int Stride = 1; Stride < BLOCK_DIM; Stride *= 2)
    {
        unsigned int Temp = 0;
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

    __shared__ unsigned int PreviousSum;
    if (threadIdx.x == 0)
    {
        unsigned int LocalSum = Shared[BLOCK_DIM - 1];

        // Wait for previous flag
        while (atomicAdd(&Flags[Bid], 0) == 0)
        {
        }

        // Read previous partial sum
        PreviousSum = ScanValue[Bid];

        // Propagate partial sum
        ScanValue[Bid + 1] = PreviousSum + LocalSum;
        __threadfence();

        // Set flag
        atomicAdd(&Flags[Bid + 1], 1);
    }
    __syncthreads();

    if (Tid < (N + 1)) // Actually need to compute number of zeros
    {
        if (Tx == 0)
        {
            Output[Tid] = PreviousSum; // First element gets previous block's sum
        }
        else
        {
            Output[Tid] = Shared[Tx - 1] + PreviousSum; // Exclusive: shift by one position
        }
    }
}

template <int BLOCK_DIM>
__global__ void ScatterValues(const unsigned int *XS, unsigned int *ExclusiveSumResult, unsigned int *Output, int Bit,
                              int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < N)
    {
        // NOTE(luatil): This could be a shared value
        unsigned int NumberOfOnes = ExclusiveSumResult[N];
        unsigned int NumberOfZeros = N - NumberOfOnes;
        unsigned int OnesBefore = ExclusiveSumResult[Tid];
        unsigned int Target = ((XS[Tid] >> Bit) & 1) ? OnesBefore + NumberOfZeros : Tid - OnesBefore;
        Output[Target] = XS[Tid];
    }
}

template <int BLOCK_DIM = 256> static void ExclusiveSum(const unsigned int *XS, unsigned int *Output, int N)
{
    dim3 BlockDim(BLOCK_DIM);
    dim3 GridDim(((N + 1) + BLOCK_DIM - 1) / BLOCK_DIM); // Have to compute number of zeros

    unsigned int *Flags, *ScanValue, *BlockCounter, *ExclusiveSumResult;
    unsigned int *Input;

    CUDA_CHECK(cudaMalloc(&Flags, sizeof(unsigned int) * (GridDim.x + 1)));
    CUDA_CHECK(cudaMalloc(&ScanValue, sizeof(unsigned int) * (GridDim.x + 1)));
    CUDA_CHECK(cudaMalloc(&BlockCounter, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ExclusiveSumResult, sizeof(unsigned int) * (N + 1))); // Have to compute number of zeros
    CUDA_CHECK(cudaMalloc(&Input, sizeof(unsigned int) * N));
    CUDA_CHECK(cudaMemcpy(Input, XS, sizeof(unsigned int) * N, cudaMemcpyDeviceToDevice));

    for (int Bit = 0; Bit < 32; Bit++)
    {
        // Initialize all of the memory
        CUDA_CHECK(cudaMemset(Flags, 0, sizeof(unsigned int) * (GridDim.x + 1)));
        CUDA_CHECK(cudaMemset(ScanValue, 0, sizeof(unsigned int) * (GridDim.x + 1)));
        CUDA_CHECK(cudaMemset(BlockCounter, 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(ExclusiveSumResult, 0, sizeof(unsigned int) * (N + 1)));

        // Set first flag to 1 to allow first block to proceed
        unsigned int One = 1;
        CUDA_CHECK(cudaMemcpy(&Flags[0], &One, sizeof(unsigned int), cudaMemcpyHostToDevice));

        ExclusiveSumKernel<BLOCK_DIM>
            <<<GridDim, BlockDim>>>(Input, ExclusiveSumResult, BlockCounter, Flags, ScanValue, Bit, N);
        ScatterValues<BLOCK_DIM><<<GridDim, BlockDim>>>(Input, ExclusiveSumResult, Output, Bit, N);

        Swap(&Input, &Output);
    }

    // After 32 iterations, result is in Input, copy to Output
    CUDA_CHECK(cudaMemcpy(Output, Input, sizeof(unsigned int) * N, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(ExclusiveSumResult));
    CUDA_CHECK(cudaFree(Input));
    CUDA_CHECK(cudaFree(Flags));
    CUDA_CHECK(cudaFree(ScanValue));
    CUDA_CHECK(cudaFree(BlockCounter));
}

int main()
{
    unsigned int Num = 0, LineIter = 0;
    while (scanf("%u\n", &Num) != EOF)
    {
        if (LineIter >= MAX_VALUE)
        {
            fprintf(stderr, "Maximum value of lines [%d] reached\n", MAX_VALUE);
            exit(1);
        }
        Lines[LineIter++] = Num;
    }

    if (LineIter == 0)
    {
        printf("No input data\n");
        return 0;
    }

    unsigned int *DeviceLines, *Output;
    CUDA_CHECK(cudaMalloc(&DeviceLines, sizeof(unsigned int) * LineIter));
    CUDA_CHECK(cudaMalloc(&Output, sizeof(unsigned int) * LineIter)); // Have to reserve space for number of zeros
    CUDA_CHECK(cudaMemcpy(DeviceLines, Lines, sizeof(unsigned int) * LineIter, cudaMemcpyHostToDevice));

    ExclusiveSum(DeviceLines, Output, LineIter);

    CUDA_CHECK(cudaMemcpy(Lines, Output, sizeof(unsigned int) * LineIter, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(DeviceLines));
    CUDA_CHECK(cudaFree(Output));

    for (unsigned int I = 0; I < LineIter; I++)
    {
        // printf("[%3d]: %d\n", I, Lines[I]);
        printf("%u\n", Lines[I]);
    }

    return 0;
}
