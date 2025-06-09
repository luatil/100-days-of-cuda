/*
 * Day 02: Simple MatMul Kernel
 */
typedef float f32;
typedef unsigned int u32;

/*
 * C[i][j] = A[i][k] * B[k][j] for k >= 0 and k <= WidthA
 */

__global__ void MatMulKernel(f32 *InputA, f32 *InputB, f32 *Output, u32 HeightA, u32 WidthA, u32 WidthB)
{
    u32 Row = blockDim.y * blockIdx.y + threadIdx.y;
    u32 Col = blockDim.x * blockIdx.x + threadIdx.x;

    if (Row < HeightA && Col < WidthB)
    {
        for (u32 K = 0; K < WidthA; K++)
        {
            Output[Row * WidthB + Col] += InputA[Row * WidthA + K] * InputB[K * WidthB + Col];
        }
    }
}
