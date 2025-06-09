/*
 * Day 07: Tiled MatMul Kernel
 *
 * Based on chapter 5 from PMPP.
 *
 */
#include <cuda_runtime.h>
#include <stdlib.h>

typedef float f32;
typedef unsigned char u8;
typedef unsigned int u32;
typedef int s32;

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

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
        u32 IndexB = (I * TILE_WIDTH + Ty) * WidthB + Col;

        TileA[Ty][Tx] = (Row < HeightA && I * TILE_WIDTH + Tx < WidthA) ? InputA[IndexA] : 0.0f;
        TileB[Ty][Tx] = (I * TILE_WIDTH + Ty < WidthA && Col < WidthB) ? InputB[IndexB] : 0.0f;

        __syncthreads();

        for (u32 K = 0; K < TILE_WIDTH; K++)
        {
            DotProduct += TileA[Ty][K] * TileB[K][Tx];
        }

        __syncthreads();
    }

    if (Row < HeightA && Col < WidthB)
    {
        Output[Row * WidthB + Col] = DotProduct;
    }
}
