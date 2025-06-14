typedef float f32;
typedef unsigned int u32;

__global__ void AddKernelCoarsened2(f32 *InA, f32 *InB, f32 *Out, u32 Length)
{
    u32 Tid = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
    if (Tid < Length)
    {
        Out[Tid] = InA[Tid] + InB[Tid];
    }
    if (Tid + 1 < Length)
    {
        Out[Tid + 1] = InA[Tid + 1] + InB[Tid + 1];
    }
}
