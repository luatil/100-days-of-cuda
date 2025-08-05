#include <cuda_runtime.h>
#include <stdlib.h>

typedef float f32;
typedef unsigned int u32;

__global__ void AddVectorFloat4(const f32 *A, const f32 *B, f32 *C, u32 N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (Tid * 4 < N)
    {
        float4 A = reinterpret_cast<const float4 *>(A)[Tid];
        float4 B = reinterpret_cast<const float4 *>(B)[Tid];
        float4 C = make_float4(A.x + B.x, A.y + B.y, A.z + B.z, A.w + B.w);
        reinterpret_cast<float4 *>(C)[Tid] = C;
    }
}
