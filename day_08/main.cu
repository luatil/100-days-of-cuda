/*
 * Day 08: Random Matrix Generation
 */
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef float f32;
typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int32_t s32;

#define AllocateCPU(_Type, _NumberOfElements) ((_Type *)malloc(sizeof(_Type) * (_NumberOfElements)))

#ifndef DEBUG_ENABLED
#define DEBUG_ENABLED 1
#endif

#if DEBUG_ENABLED
#define DbgU32(_Val) printf(#_Val "=%d\n", (_Val))
#define DbgS32(_Val) printf(#_Val "=%d\n", (_Val))
#define DbgF32(_Val) printf(#_Val "=%f\n", (_Val))
#else
#define DbgU32(_Val)
#define DbgS32(_Val)
#define DbgF32(_Val)
#endif

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void GenerateRandomMatrix(f32 *OutputMatrix, u32 Width, u32 Height, u64 Seed)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    int TotalElements = Width * Height;

    if (Tid < TotalElements)
    {
        curandState State;
        curand_init(Seed, Tid, 0, &State);
        OutputMatrix[Tid] = curand_uniform(&State); // Uniform [0,1]
    }
}

int main()
{
    u32 N = 10;
    u32 M = 10;
    u32 SizeInBytes = sizeof(f32) * N * M;
    u64 Seed = 2345;

    f32 *HostMatrix= AllocateCPU(f32, N * M);

    f32 *DeviceMatrix;
    cudaMalloc(&DeviceMatrix, SizeInBytes);

    dim3 ThreadsPerBlock(32);
    dim3 BlocksPerGrid((N*M + 32 - 1) / 32);

    GenerateRandomMatrix<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceMatrix, N, M, Seed);

    cudaMemcpy(HostMatrix, DeviceMatrix, SizeInBytes, cudaMemcpyDeviceToHost);

    fprintf(stdout, "%d %d\n", N, M);
    for (u32 I = 0; I < N; I++) 
    {
	for (u32 J = 0; J < M; J++)
	{
	    fprintf(stdout, "%.6f ", HostMatrix[I*N+J]);
	}
	fprintf(stdout, "\n");
    }
}
