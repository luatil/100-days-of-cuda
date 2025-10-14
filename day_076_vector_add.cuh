#ifndef DAY_076_VECTOR_ADD_CUH
#define DAY_076_VECTOR_ADD_CUH

static float Abs(float X)
{
  if (X < 0) return -X;
  return X;
}

__global__ void VectorAdd(float *A, float *B, float *C, int N)
{
  int Tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (Tid < N)
  {
    C[Tid] = A[Tid] + B[Tid];
  }
}

#endif // DAY_076_VECTOR_ADD_CUH
