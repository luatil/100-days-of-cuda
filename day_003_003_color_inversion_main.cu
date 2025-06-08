/*
 * Day 03: Color inversion
 *
 * Idea from leetgpu. Make it a command line utility with stbimage.h
 */
#include <stdio.h>
#include <stdlib.h>

#pragma nv_diag_push
#pragma nv_diag_suppress 550
#define STB_IMAGE_IMPLEMENTATION
#include "third/stb_image.h"
#pragma nv_diag_pop

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third/stb_image_write.h"

typedef float f32;

typedef unsigned char u8;
typedef unsigned int u32;

typedef int s32;

#define AllocateCPU(_Type, _NumberOfElements)                                  \
  (_Type *)malloc(sizeof(_Type) * (_NumberOfElements))

#ifndef DEBUG_ENABLED
#define DEBUG_ENABLED 1
#endif

#if DEBUG_ENABLED
#define DbgU32(_Val) printf(#_Val "=%d\n", (_Val))
#define DbgS32(_Val) printf(#_Val "=%d\n", (_Val))
#define DbgF32(_Val) printf(#_Val "=%f\n", (_Val))
#else
#define DbgU32(_Val)
#define DbgS32(_Val)
#define DbgF32(_Val)
#endif

/*
 * I[Tid] = 255 - I[Tid] if Tid % 4 != 0
 * Invert all channels but alpha.
 */
__global__ void InvertImage(u8 *ImageBuffer, u32 SizeInBytes) {
  u32 Tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (Tid < SizeInBytes && (Tid % 4 != 0)) {
      ImageBuffer[Tid] = 255 - ImageBuffer[Tid];
  }
}

int main(int ArgumentCount, char **Arguments) {

  if (ArgumentCount < 3) {
    fprintf(stderr, "%s <input_filename> <output_filename>\n", Arguments[0]);
    exit(1);
  }

  char *InputFilename = Arguments[1];
  char *OutputFilename = Arguments[2];

  s32 Width, Height, ChannelsInOriginalFile;

  // Output will always have the last number of channels from 1..4 as valid options.
  u8 *ImageBuffer =
      stbi_load(InputFilename, &Width, &Height, &ChannelsInOriginalFile, 4); // Force RGBA

  if (!ImageBuffer) {
      fprintf(stderr, "Image not valid. Exiting program.\n");
      exit(1);
  }

  DbgS32(Width);
  DbgS32(Height);
  DbgS32(ChannelsInOriginalFile);

  // Allocate the gpu memory
  u32 SizeInBytes = Width * Height * 4;
  u8 *DeviceImageBuffer;
  cudaMalloc(&DeviceImageBuffer, SizeInBytes);

  // Transfer from Host to Device
  cudaMemcpy(DeviceImageBuffer, ImageBuffer, SizeInBytes, cudaMemcpyHostToDevice);

  // Launch Kernel to invert the color of the image

  // I think I want to have each warp operate on contiguous data. Which
  // means that each channel get's it's own thread.  And I just discard
  // data if comes from the Alpha channel. Should benchmark this intuition
  // later
  dim3 ThreadsPerBlock(256);
  dim3 BlocksPerGrid((SizeInBytes + 256 - 1) / 256);

  InvertImage<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceImageBuffer, SizeInBytes);

  // Read from Device back to host. Should I do inplace?
  cudaMemcpy(ImageBuffer, DeviceImageBuffer, SizeInBytes, cudaMemcpyDeviceToHost);

  // Write the result to output file.
  stbi_write_jpg(OutputFilename, Width, Height, 4, ImageBuffer, 100);

  printf("DAY_03: CUDA SUCCESS");
}
