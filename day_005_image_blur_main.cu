/*
 * Day 05: Image Blur
 *
 * Based on chapter 3 from PMPP. Handles 1 channel images.
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
#include "day_005_image_blur_kernel.cu"

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
    s32 DesiredChannels = 1;

    // Output will always have the last number of channels from 1..4 as valid
    // options.
    u8 *InputImage = stbi_load(InputFilename, &Width, &Height, &ChannelsInOriginalFile, DesiredChannels);

    u32 SizeInBytes = Width * Height * DesiredChannels;

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
    cudaMalloc(&DeviceInputImage, SizeInBytes);
    cudaMalloc(&DeviceOutputImage, SizeInBytes);

    // Transfer from Host to Device
    cudaMemcpy(DeviceInputImage, InputImage, SizeInBytes, cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(16, 16);
    dim3 BlocksPerGrid((Width + 16 - 1) / 16, (Height + 16 - 1) / 16);

    BlurImage<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceInputImage, DeviceOutputImage, Width, Height);

    u8 *OutputImage = AllocateCPU(u8, SizeInBytes);
    cudaMemcpy(OutputImage, DeviceOutputImage, SizeInBytes, cudaMemcpyDeviceToHost);

    // Write the result to output file.
    stbi_write_jpg(OutputFilename, Width, Height, DesiredChannels, OutputImage, 100);

    printf("DAY_05: CUDA SUCCESS");
}
