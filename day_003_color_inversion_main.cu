/*
 * Day 03: Color inversion
 *
 * Idea from leetgpu. Make it a command line utility with stbimage.h
 */
#include <stdio.h>
#include <stdlib.h>

typedef float f32;

typedef unsigned char u8;
typedef unsigned int u32;

typedef int s32;

#include "day_001_macros.h"
#include "day_003_color_inversion_kernel.cu"
#include "day_003_vendor_libraries.h"

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

    // Output will always have the last number of channels from 1..4 as valid options.
    u8 *ImageBuffer = stbi_load(InputFilename, &Width, &Height, &ChannelsInOriginalFile, 4); // Force RGBA

    if (!ImageBuffer)
    {
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
