/*
 * Day 05: Image Blur
 *
 * Based on chapter 3 from PMPP. Handles 1 channel images.
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

#ifndef BLUR_SIZE
#define BLUR_SIZE 10
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
 * Receives a 1 channel image in InputImage and writes a
 * blurred version of the image to OutputImage.
 *
 * Both have only 1 channel.
 *
 * For each output pixel P[x][y] we have
 *
 * P[x][y] = average for i, j in [-BLUR_SIZE,BLUR_SIZE] of I[x+i][y+j]
 *
 * where I[x][y] represents the input image.
 */
__global__ void BlurImage(const u8 *InputImage, u8 *OutputImage, u32 Width,
                          u32 Height) {
  u32 Row = blockDim.y * blockIdx.y + threadIdx.y;
  u32 Col = blockDim.x * blockIdx.x + threadIdx.x;

  if ((Row < Height) && (Col < Width)) {

    u32 PixelValue = 0;
    u32 Pixels = 0;

    for (s32 I = -BLUR_SIZE; I <= BLUR_SIZE; I++) {
      for (s32 J = -BLUR_SIZE; J <= BLUR_SIZE; J++) {
        u32 TargetRow = Row + I;
        u32 TargetCol = Col + J;

        if (TargetRow > 0 && TargetRow < Height && TargetCol > 0 &&
            TargetCol < Width) {
          PixelValue += InputImage[TargetRow * Width + TargetCol];
          Pixels++;
        }
      }
    }
    OutputImage[Row * Width + Col] = PixelValue / Pixels;
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
  s32 DesiredChannels = 1;

  // Output will always have the last number of channels from 1..4 as valid
  // options.
  u8 *InputImage = stbi_load(InputFilename, &Width, &Height,
                             &ChannelsInOriginalFile, DesiredChannels);

  u32 SizeInBytes = Width * Height * DesiredChannels;

  if (!InputImage) {
    fprintf(stderr, "Image not valid. Exiting program.\n");
    exit(1);
  }

  DbgS32(Width);
  DbgS32(Height);
  DbgS32(ChannelsInOriginalFile);

  // Allocate the gpu memory
  u8 *DeviceInputImage, *DeviceOutputImage;
  cudaMalloc(&DeviceInputImage, SizeInBytes);
  cudaMalloc(&DeviceOutputImage, SizeInBytes);

  // Transfer from Host to Device
  cudaMemcpy(DeviceInputImage, InputImage, SizeInBytes, cudaMemcpyHostToDevice);

  dim3 ThreadsPerBlock(16, 16);
  dim3 BlocksPerGrid((Width + 16 - 1) / 16, (Height + 16 - 1) / 16);

  BlurImage<<<BlocksPerGrid, ThreadsPerBlock>>>(
      DeviceInputImage, DeviceOutputImage, Width, Height);

  u8 *OutputImage = AllocateCPU(u8, SizeInBytes);
  cudaMemcpy(OutputImage, DeviceOutputImage, SizeInBytes,
             cudaMemcpyDeviceToHost);

  // Write the result to output file.
  stbi_write_jpg(OutputFilename, Width, Height, DesiredChannels, OutputImage,
                 100);

  printf("DAY_05: CUDA SUCCESS");
}
