/*
 * Day 03: Color inversion
 *
 * Idea from leetgpu. Make it a command line utility with stbimage.h
 */

typedef unsigned char u8;
typedef unsigned int u32;

/*
 * I[Tid] = 255 - I[Tid] if Tid % 4 != 0
 * Invert all channels but alpha.
 */
__global__ void InvertImage(u8 *ImageBuffer, u32 SizeInBytes)
{
    u32 Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < SizeInBytes && (Tid % 4 != 0))
    {
        ImageBuffer[Tid] = 255 - ImageBuffer[Tid];
    }
}
