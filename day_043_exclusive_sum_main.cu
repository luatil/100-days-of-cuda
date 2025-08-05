/*
 * NAME
 * 	esum - calculates exclusive sum
 *
 * SYNOPSIS
 * 	esum
 *
 * DESCRIPTION
 * 	Calculates the exclusive sum from numbers from stdin and prints
 * 	the result to stdout.
 *
 * USAGE
 * 	Calculate the exclusive sum from the first 10 natural numbers
 * 	- seq 1 10 | esum
 *
 */
#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_VALUE (1 << 24)
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

static int Lines[MAX_VALUE];

template <int BLOCK_DIM>
__global__ void ExclusiveSumKernel(int *XS, int *BlockCounter, int *Flags, int *ScanValue, int N)
{
    // Dynamically select BlockId based on scheduling order
    __shared__ unsigned int BidS;
    if (threadIdx.x == 0)
    {
        BidS = atomicAdd(BlockCounter, 1);
    }
    __syncthreads();
    unsigned int Bid = BidS;

    __shared__ int Shared[BLOCK_DIM];

    // For now no thread coarsening
    int Tid = blockDim.x * Bid + threadIdx.x;
    int Tx = threadIdx.x;

    Shared[Tx] = Tid < N ? XS[Tid] : 0;
    __syncthreads();

    // Perform inclusive scan within block
    for (int Stride = 1; Stride < BLOCK_DIM; Stride *= 2)
    {
        int Temp = 0;
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

    __shared__ int PreviousSum;
    if (threadIdx.x == 0)
    {
        int LocalSum = Shared[BLOCK_DIM - 1];

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

    if (Tid < N)
    {
        if (Tx == 0)
        {
            XS[Tid] = PreviousSum; // First element gets previous block's sum
        }
        else
        {
            XS[Tid] = Shared[Tx - 1] + PreviousSum; // Exclusive: shift by one position
        }
    }
}

template <int BLOCK_DIM = 256> static void ExclusiveSum(int *XS, int N)
{
    dim3 BlockDim(BLOCK_DIM);
    dim3 GridDim((N + BLOCK_DIM - 1) / BLOCK_DIM);

    int *Flags, *ScanValue, *BlockCounter;

    CUDA_CHECK(cudaMalloc(&Flags, sizeof(int) * (GridDim.x + 1)));
    CUDA_CHECK(cudaMalloc(&ScanValue, sizeof(int) * (GridDim.x + 1)));
    CUDA_CHECK(cudaMalloc(&BlockCounter, sizeof(int)));

    // Initialize all of the memory
    CUDA_CHECK(cudaMemset(Flags, 0, sizeof(int) * (GridDim.x + 1)));
    CUDA_CHECK(cudaMemset(ScanValue, 0, sizeof(int) * (GridDim.x + 1)));
    CUDA_CHECK(cudaMemset(BlockCounter, 0, sizeof(int)));

    // Set first flag to 1 to allow first block to proceed
    int One = 1;
    CUDA_CHECK(cudaMemcpy(&Flags[0], &One, sizeof(int), cudaMemcpyHostToDevice));

    ExclusiveSumKernel<BLOCK_DIM><<<GridDim, BlockDim>>>(XS, BlockCounter, Flags, ScanValue, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(Flags));
    CUDA_CHECK(cudaFree(ScanValue));
    CUDA_CHECK(cudaFree(BlockCounter));
}

int main()
{
    int Num = 0, LineIter = 0;
    while (scanf("%d\n", &Num) != EOF)
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

    int *DeviceLines;
    CUDA_CHECK(cudaMalloc(&DeviceLines, sizeof(int) * LineIter));
    CUDA_CHECK(cudaMemcpy(DeviceLines, Lines, sizeof(int) * LineIter, cudaMemcpyHostToDevice));

    ExclusiveSum(DeviceLines, LineIter);

    CUDA_CHECK(cudaMemcpy(Lines, DeviceLines, sizeof(int) * LineIter, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(DeviceLines));

    for (int I = 0; I < LineIter; I++)
    {
        printf("[%3d]: %d\n", I, Lines[I]);
    }

    return 0;
}
