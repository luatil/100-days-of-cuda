/*
 * Day 05: Image Blur
 *
 * Based on chapter 3 from PMPP. Handles 1 channel images.
 *
 * Make it a command line utility with stbimage.h
 */
#include <stdlib.h>

typedef unsigned char u8;
typedef unsigned int u32;

#ifndef BLUR_SIZE
#define BLUR_SIZE 10
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
__global__ void BlurImage(const u8 *InputImage, u8 *OutputImage, u32 Width, u32 Height)
{
    u32 Row = blockDim.y * blockIdx.y + threadIdx.y;
    u32 Col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((Row < Height) && (Col < Width))
    {

        u32 PixelValue = 0;
        u32 Pixels = 0;

        for (int I = -BLUR_SIZE; I <= BLUR_SIZE; I++)
        {
            for (int J = -BLUR_SIZE; J <= BLUR_SIZE; J++)
            {
                u32 TargetRow = Row + I;
                u32 TargetCol = Col + J;

                if (TargetRow > 0 && TargetRow < Height && TargetCol > 0 && TargetCol < Width)
                {
                    PixelValue += InputImage[TargetRow * Width + TargetCol];
                    Pixels++;
                }
            }
        }
        OutputImage[Row * Width + Col] = PixelValue / Pixels;
    }
}
