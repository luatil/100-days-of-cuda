/*
 * Day 04: Grayscale Conversion
 *
 * Based on chapter 3 from PMPP.
 * Main difference is that it takes a RGBA image instead of a RGA one.
 *
 * Make it a command line utility with stbimage.h
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
 * Receives a RGBA image in InputImage and write a single
 * channel to OutputImage.
 *
 * L = 0.21*r + 0.72*g + 0.07b
 *
 * Where L represents the result of the single channel
 * and rgb follows the normal convention.
 */
__global__ void ConvertToGrayscale(const u8 *InputImage, u8 *OutputImage,
                                   u32 Width, u32 Height) {
  u32 Row = blockDim.y * blockIdx.y + threadIdx.y;
  u32 Col = blockDim.x * blockIdx.x + threadIdx.x;

  if ((Row < Height) && (Col < Width)) {
    u32 Index = Row * Width + Col;
    u8 R = InputImage[4 * Index + 0];
    u8 G = InputImage[4 * Index + 1];
    u8 B = InputImage[4 * Index + 2];
    OutputImage[Index] = 0.21 * R + 0.72 * G + 0.07 * B;
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

  // Output will always have the last number of channels from 1..4 as valid
  // options.
  u8 *InputImage = stbi_load(InputFilename, &Width, &Height,
                             &ChannelsInOriginalFile, 4); // Force RGBA

  if (!InputImage) {
    fprintf(stderr, "Image not valid. Exiting program.\n");
    exit(1);
  }

  DbgS32(Width);
  DbgS32(Height);
  DbgS32(ChannelsInOriginalFile);

  // Allocate the gpu memory
  u8 *DeviceInputImage, *DeviceOutputImage;
  cudaMalloc(&DeviceInputImage, Width * Height * 4);
  cudaMalloc(&DeviceOutputImage, Width * Height);

  // Transfer from Host to Device
  cudaMemcpy(DeviceInputImage, InputImage, Width * Height * 4,
             cudaMemcpyHostToDevice);

  dim3 ThreadsPerBlock(16, 16);
  dim3 BlocksPerGrid((Width + 16 - 1) / 16, (Height + 16 - 1) / 16);

  ConvertToGrayscale<<<BlocksPerGrid, ThreadsPerBlock>>>(
      DeviceInputImage, DeviceOutputImage, Width, Height);

  u8 *OutputImage = AllocateCPU(u8, Width * Height);
  cudaMemcpy(OutputImage, DeviceOutputImage, Width * Height,
             cudaMemcpyDeviceToHost);

  // Write the result to output file.
  stbi_write_jpg(OutputFilename, Width, Height, 1, OutputImage, 100);

  printf("DAY_04: CUDA SUCCESS");
}
