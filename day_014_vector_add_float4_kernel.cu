#include <cuda_runtime.h>
#include <stdlib.h>

typedef float f32;
typedef unsigned int u32;

__global__ void AddVector_Float4(const f32 *A, const f32 *B, f32 *C, u32 N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (Tid * 4 < N)
    {
        float4 a = reinterpret_cast<const float4 *>(A)[Tid];
        float4 b = reinterpret_cast<const float4 *>(B)[Tid];
        float4 c = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        reinterpret_cast<float4 *>(C)[Tid] = c;
    }
}
