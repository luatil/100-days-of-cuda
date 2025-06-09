/*
 * Day 001: Simple VecAdd Kernel
 */

typedef float f32;
typedef unsigned int u32;

__global__ void AddKernel(f32 *InputA, f32 *InputB, f32 *Output, u32 Length)
{
    u32 Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < Length)
    {
        Output[Tid] = InputA[Tid] + InputB[Tid];
    }
}
