#include <cuda_runtime.h>
#include <stdlib.h>

typedef float f32;
typedef unsigned int u32;

__global__ void AddVectorFloat4(f32 *A, f32 *B, f32 *C, u32 N)
{
    int Tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (Tid * 4 < N)
    {
	float4 A4 = make_float4(A[Tid], A[Tid + 1], A[Tid+2], A[Tid+3]);
	float4 B4 = make_float4(B[Tid], B[Tid + 1], B[Tid+2], B[Tid+3]);
        float4 C4 = make_float4(A4.x + B4.x, A4.y + B4.y, A4.z + B4.z, A4.w + B4.w);
        reinterpret_cast<float4 *>(C)[Tid] = C4;
    }
}
