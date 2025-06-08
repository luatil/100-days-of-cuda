/*
 * Day 07: Tiled MatMul Kernel
 *
 * Based on chapter 5 from PMPP.
 *
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

typedef float f32;
typedef unsigned char u8;
typedef unsigned int u32;
typedef int s32;

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

static f32 Eps = 1e-6;

/*
 * C[i][j] = A[i][k] * B[k][j] for k >= 0 and k <= WidthA
 */

__global__ void TiledMatmulKernel(f32 *InputA, f32 *InputB, f32 *Output, u32 HeightA, u32 WidthA, u32 WidthB)
{

    __shared__ f32 TileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ f32 TileB[TILE_WIDTH][TILE_WIDTH];

    u32 Row = blockDim.y * blockIdx.y + threadIdx.y;
    u32 Col = blockDim.x * blockIdx.x + threadIdx.x;

    u32 Tx = threadIdx.x;
    u32 Ty = threadIdx.y;

    f32 DotProduct = 0.0f;

    for (u32 I = 0; I < (WidthA + TILE_WIDTH - 1) / TILE_WIDTH; I++)
    {

        u32 IndexA = Row * WidthA + I * TILE_WIDTH + Tx;
        u32 IndexB = Col + Ty * WidthB + I * TILE_WIDTH * WidthB;

        TileA[Tx][Ty] = (IndexA < (HeightA * WidthA)) ? InputA[IndexA] : 0.0f;
        TileB[Tx][Ty] = (IndexB < (WidthA * WidthB)) ? InputB[IndexB] : 0.0f;

        __syncthreads();

        for (u32 K = 0; K < TILE_WIDTH; K++)
        {
            DotProduct += TileA[Tx][K] * TileB[K][Ty];
        }

        __syncthreads();
    }

    if (Row < HeightA && Col < WidthB)
    {
        Output[Row * WidthB + Col] = DotProduct;
    }
}

int main()
{
    u32 N = 256;
    u32 M = 256;
    u32 SizeInBytes = sizeof(f32) * N * M;

    f32 *HostA = AllocateCPU(f32, N * M);
    f32 *HostB = AllocateCPU(f32, N * M);
    f32 *HostC = AllocateCPU(f32, N * M);

    for (u32 I = 0; I < (N * M); I++)
    {
        HostA[I] = 1.0f;
        HostB[I] = 2.0f;
    }

    f32 *DeviceA, *DeviceB, *DeviceC;
    cudaMalloc(&DeviceA, SizeInBytes);
    cudaMalloc(&DeviceB, SizeInBytes);
    cudaMalloc(&DeviceC, SizeInBytes);

    cudaMemcpy(DeviceA, HostA, SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(DeviceB, HostB, SizeInBytes, cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(16, 16, 1);
    dim3 BlocksPerGrid((N + 16 - 1) / 16, (M + 16 - 1) / 16, 1);

    TiledMatmulKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceA, DeviceB, DeviceC, N, N, N);

    cudaMemcpy(HostC, DeviceC, SizeInBytes, cudaMemcpyDeviceToHost);

    for (u32 I = 0; I < (N * M); I++)
    {
        f32 Exp = 2.0f * 256.0f;
        f32 Diff = HostC[I] - Exp;
        if (abs(Diff) > Eps)
        {
            printf("Cuda Kernel Failed | Pos: %d | Expected %f Got %f", I, Exp, HostC[I]);
            exit(1);
        }
    }

    printf("DAY_07: CUDA SUCCESS");
}
