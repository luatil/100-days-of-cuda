/*
 * Day 02: Simple MatMul Kernel
 */
#include <stdio.h>
#include <stdlib.h>

typedef float f32;
typedef unsigned int u32;

#define AllocateCPU(_Type, _NumberOfElements)                                  \
  (_Type *)malloc(sizeof(_Type) * (_NumberOfElements))

#ifndef DEBUG_ENABLED
#define DEBUG_ENABLED 1
#endif

#if DEBUG_ENABLED
#define DbgU32(_Val) printf(#_Val "=%d\n", (_Val))
#define DbgF32(_Val) printf(#_Val "=%f\n", (_Val))
#else
#define DbgU32(_Val)
#define DbgF32(_Val)
#endif

static f32 Eps = 1e-6;

/*
 * C[i][j] = A[i][k] * B[k][j] for k >= 0 and k <= WidthA
 */

__global__ void MatMulKernel(f32 *InputA, f32 *InputB, f32 *Output, u32 HeightA,
                             u32 WidthA, u32 WidthB) {
  u32 Row = blockDim.y * blockIdx.y + threadIdx.y;
  u32 Col = blockDim.x * blockIdx.x + threadIdx.x;

  if (Row < HeightA && Col < WidthB) {
    for (u32 K = 0; K < WidthA; K++) {
      Output[Row * WidthB + Col] +=
          InputA[Row * WidthA + K] * InputB[K * WidthB + Row];
    }
  }
}

int main() {
  u32 N = 256;
  u32 M = 256;
  u32 SizeInBytes = sizeof(f32) * N * M;

  f32 *HostA = AllocateCPU(f32, N * M);
  f32 *HostB = AllocateCPU(f32, N * M);
  f32 *HostC = AllocateCPU(f32, N * M);

  for (u32 I = 0; I < (N * M); I++) {
    HostA[I] = 1.0f;
    HostB[I] = 2.0f;
  }

  f32 *DeviceA, *DeviceB, *DeviceC;
  cudaMalloc(&DeviceA, SizeInBytes);
  cudaMalloc(&DeviceB, SizeInBytes);
  cudaMalloc(&DeviceC, SizeInBytes);

  cudaMemcpy(DeviceA, HostA, SizeInBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(DeviceB, HostB, SizeInBytes, cudaMemcpyHostToDevice);

  dim3 ThreadsPerBlock(16, 16, 1);
  dim3 BlocksPerGrid(256 / 16, 256 / 16, 1);

  MatMulKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceA, DeviceB, DeviceC, N,
                                                   N, N);

  cudaMemcpy(HostC, DeviceC, SizeInBytes, cudaMemcpyDeviceToHost);

  for (u32 I = 0; I < (N * M); I++) {
    f32 Exp = 2.0f * 256.0f;
    f32 Diff = HostC[I] - Exp;
    if (abs(Diff) > Eps) {
      printf("Cuda Kernel Failed | Pos: %d | Expected %f Got %f", I, Exp,
             HostC[I]);
      exit(1);
    }
  }

  printf("DAY_02: CUDA SUCCESS");
}
