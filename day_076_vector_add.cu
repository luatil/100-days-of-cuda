#include "day_076_vector_add.cuh"

__global__ void VectorAdd(float *A, float *B, float *C, int N)
{
  int Tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (Tid < N)
  {
    C[Tid] = A[Tid] + B[Tid];
  }
}
