#include <stdio.h>
#include <stdlib.h>

typedef float f32;
typedef unsigned int u32;

#define AllocateCPU(_Type, _NumberOfElements)                                  \
  (_Type *)malloc(sizeof(_Type) * _NumberOfElements)

#ifndef DEBUG_ENABLED
#define DEBUG_ENABLED 0
#endif

#if DEBUG_ENABLED
#define DbgU32(_Val) printf(#_Val "=%d\n", (_Val))
#define DbgF32(_Val) printf(#_Val "=%f\n", (_Val))
#else
#define DbgU32(_Val)
#define DbgF32(_Val)
#endif

static f32 Eps = 1e-6;

__global__ void AddKernel(f32 *InputA, f32 *InputB, f32 *Output, u32 Length) {
  u32 Tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (Tid < Length) {
    Output[Tid] = InputA[Tid] + InputB[Tid];
  }
}

int main() {
  u32 N = 4096;
  u32 SizeInBytes = sizeof(f32) * N;

  f32 *HostA = AllocateCPU(f32, N);
  f32 *HostB = AllocateCPU(f32, N);
  f32 *HostC = AllocateCPU(f32, N);

  for (u32 I = 0; I < N; I++) {
    HostA[I] = 1.0f;
    HostB[I] = 2.0f;
  }

  f32 *DeviceA, *DeviceB, *DeviceC;
  cudaMalloc(&DeviceA, SizeInBytes);
  cudaMalloc(&DeviceB, SizeInBytes);
  cudaMalloc(&DeviceC, SizeInBytes);

  cudaMemcpy(DeviceA, HostA, SizeInBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(DeviceB, HostB, SizeInBytes, cudaMemcpyHostToDevice);

  u32 NumberOfThreads = 32;
  u32 NumberOfBlocks = (N + NumberOfThreads - 1) / NumberOfThreads;

  DbgU32(NumberOfThreads);
  DbgU32(NumberOfBlocks);
  DbgU32(NumberOfThreads * NumberOfBlocks);

  dim3 Dim3NumberOfThreads(NumberOfThreads);
  dim3 Dim3NumberOfBlocks(NumberOfBlocks);

  AddKernel<<<Dim3NumberOfBlocks, Dim3NumberOfThreads>>>(DeviceA, DeviceB,
                                                         DeviceC, N);

  cudaMemcpy(HostC, DeviceC, SizeInBytes, cudaMemcpyDeviceToHost);

  for (u32 I = 0; I < N; I++) {
    f32 Diff = HostC[I] - 3.0f;
    if (abs(Diff) > Eps) {
      printf("Cuda Kernel Failed | Pos: %d | Expected %f Got %f", I, 3.0f,
             HostC[I]);
      exit(1);
    }
  }

  printf("DAY_01: CUDA SUCCESS");
}
