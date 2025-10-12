#include <stdio.h>

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


int main()
{
  cudaError_t err = cudaSuccess;
  const int N = 1024;
  float *A = (float*)malloc(sizeof(float)*N);
  float *B = (float*)malloc(sizeof(float)*N);
  float *C = (float*)malloc(sizeof(float)*N);

  for (int I = 0; I < N; I++)
  {
    A[I] = 1.0f;
    B[I] = 2.0f;
  }

  size_t size = N * sizeof(float);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }


  int BlockDim = 256;
  int GridDim = (N + BlockDim - 1) / BlockDim;

  VectorAdd<<<GridDim, BlockDim>>>(d_A, d_B, d_C, N);

  err = cudaGetLastError();

  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }


  cudaMemcpy(C, d_C, sizeof(float)*N, cudaMemcpyDeviceToHost);

  // Now we do a test
  for (int I = 0; I < N; I++)
  {
    if(Abs(C[I] - 3.0f) > 0.01) 
    {
      // printf("A[I] = %f B[I] = %f C[I] = %f\n", A[I], B[I], C[I]);
      printf("Kernel Failed\n");
      exit(1);
    }
  }

  printf("VECTOR_ADD: Pass");
}
