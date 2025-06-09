/*
 * Day 04: Grayscale Conversion
 *
 * Based on chapter 3 from PMPP.
 * Main difference is that it takes a RGBA image instead of a RGA one.
 *
 * Make it a command line utility with stbimage.h
 */

typedef unsigned char u8;
typedef unsigned int u32;

/*
 * Receives a RGBA image in InputImage and write a single
 * channel to OutputImage.
 *
 * L = 0.21*r + 0.72*g + 0.07b
 *
 * Where L represents the result of the single channel
 * and rgb follows the normal convention.
 */
__global__ void ConvertToGrayscale(const u8 *InputImage, u8 *OutputImage, u32 Width, u32 Height)
{
    u32 Row = blockDim.y * blockIdx.y + threadIdx.y;
    u32 Col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((Row < Height) && (Col < Width))
    {
        u32 Index = Row * Width + Col;
        u8 R = InputImage[4 * Index + 0];
        u8 G = InputImage[4 * Index + 1];
        u8 B = InputImage[4 * Index + 2];
        OutputImage[Index] = 0.21 * R + 0.72 * G + 0.07 * B;
    }
}
