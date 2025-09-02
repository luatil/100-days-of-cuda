#include <stdio.h>

#define ArraySize(_X) (sizeof(_X) / sizeof(_X[0]))

__global__ void CountEqualKernel(const int *Input, int *Output, int N, int K)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < N && Input[Tid] == K)
    {
        atomicAdd(Output, 1);
    }
}

int main()
{
  int Input[] = {1, 2, 3, 4, 1};
  int K = 1;

  int *DeviceInput;
  cudaMalloc(&DeviceInput, sizeof(int) * 5);
  cudaMemcpy(DeviceInput, Input, sizeof(int) * 5, cudaMemcpyHostToDevice);

  int *DeviceOutput;
  cudaMalloc(&DeviceOutput, sizeof(int));
  cudaMemset(DeviceOutput, 0, sizeof(int));

  int BlockDim = 256;
  int GridDim = ArraySize(Input);

  CountEqualKernel<<<GridDim, BlockDim>>>(DeviceInput, DeviceOutput, 5, K);

  cudaDeviceSynchronize();

  int Result;
  cudaMemcpy(&Result, DeviceOutput, sizeof(int), cudaMemcpyDeviceToHost);

  printf("Number of elements equal to %d: %d\n", K, Result);
}
