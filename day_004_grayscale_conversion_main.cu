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

typedef float f32;

typedef unsigned char u8;
typedef unsigned int u32;

typedef int s32;

#include "day_001_macros.h"
#include "day_003_vendor_libraries.h"
#include "day_004_grayscale_conversion_kernel.cu"

int main(int ArgumentCount, char **Arguments)
{

    if (ArgumentCount < 3)
    {
        fprintf(stderr, "%s <input_filename> <output_filename>\n", Arguments[0]);
        exit(1);
    }

    char *InputFilename = Arguments[1];
    char *OutputFilename = Arguments[2];

    s32 Width, Height, ChannelsInOriginalFile;

    // Output will always have the last number of channels from 1..4 as valid
    // options.
    u8 *InputImage = stbi_load(InputFilename, &Width, &Height, &ChannelsInOriginalFile, 4); // Force RGBA

    if (!InputImage)
    {
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
    cudaMemcpy(DeviceInputImage, InputImage, Width * Height * 4, cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(16, 16);
    dim3 BlocksPerGrid((Width + 16 - 1) / 16, (Height + 16 - 1) / 16);

    ConvertToGrayscale<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceInputImage, DeviceOutputImage, Width, Height);

    u8 *OutputImage = AllocateCPU(u8, Width * Height);
    cudaMemcpy(OutputImage, DeviceOutputImage, Width * Height, cudaMemcpyDeviceToHost);

    // Write the result to output file.
    stbi_write_jpg(OutputFilename, Width, Height, 1, OutputImage, 100);

    printf("DAY_04: CUDA SUCCESS");
}
