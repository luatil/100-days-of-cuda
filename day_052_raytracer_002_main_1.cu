#include <stdlib.h>

#include "day_003_vendor_libraries.h"
#include "day_051_cli.h"
#include "day_051_types.h"
#include "day_051_vector_2.cu"

template <typename T> __device__ __host__ T CircleSDF(vec2<T> P, vec2<T> C, T R)
{
    return Distance(P, C) - R;
}

template <typename T> __device__ __host__ T Abs(T X)
{
    if (X < 0)
    {
        return -X;
    }
    else
    {
        return X;
    }
}

__device__ __host__ b32 CloseF32(f32 X, f32 Y, f32 Eps = 0.01)
{
    if (Abs(X - Y) < Eps)
    {
        return 1;
    }
    return 0;
}

template <typename T> __device__ __host__ b32 CloseVec2(vec2<T> X, vec2<T> Y, f32 Eps = 0.01f)
{
    if ((X - Y).Length() < Eps)
    {
        return 1;
    }
    return 0;
}

__device__ __host__ f32 LineSegmentSDF(vec2<f32> P, vec2<f32> A, vec2<f32> B, f32 R)
{
    float H = min(1.0, max(0.0, Dot(P - A, B - A) / Dot(B - A, B - A)));

    return Length(P - A - (B - A) * H) - R;
}

__device__ __host__ vec2<f32> Screen2Point(s32 Row, s32 Col, u32 Width, u32 Height)
{
    vec2<f32> Result{0.0f, 0.0f};

    Result.X = (f32)Col / (f32)Width;
    Result.Y = (f32)Row / (f32)Height;

    return Result;
}

__global__ void DrawCircle(u8 *Image, vec2<f32> Circle, f32 Radius, u32 Width, u32 Height)
{
    s32 Row = blockDim.y * blockIdx.y + threadIdx.y;
    s32 Col = blockDim.x * blockIdx.x + threadIdx.x;

    if (Row < Height && Col < Width)
    {
        vec2<f32> Q = Screen2Point(Row, Col, Width, Height);

        f32 Val = CircleSDF(Q, Circle, Radius);
        float EdgeWidth = 0.001;
        float Alpha = 1.0f - SmoothStep(-EdgeWidth, EdgeWidth, (float)Val);

        // Image[Row * Width + Col] = (u8)(Alpha * 255) + (u8)(1.0f - Alpha) * Image[Row * Width + Col];
    }
}

__global__ void DrawLine(u8 *Image, vec2<f32> P0, vec2<f32> P1, u32 Width, u32 Height)
{
    s32 Row = blockDim.y * blockIdx.y + threadIdx.y;
    s32 Col = blockDim.x * blockIdx.x + threadIdx.x;

    if (Row < Height && Col < Width)
    {
        vec2<f32> Q = Screen2Point(Row, Col, Width, Height);
        f32 D = LineSegmentSDF(Q, P0, P1, 0.005f);
        // f32 Alpha = 1.0f - SmoothStep(0.0f, 1.0f, D);
        f32 Alpha = Clamp(D, 0.0f, 0.3f);
        Image[Row * Width + Col] = (u8)(Alpha * 255) + (u8)(1.0f - Alpha) * Image[Row * Width + Col];
    }
}

int main(int argc, char *argv[])
{
    options Opts = ParseCommandLine(argc, argv);

    u8 *d_Image;

    cudaMalloc(&d_Image, sizeof(u8) * Opts.Width * Opts.Height);

    dim3 BlockDim(16, 16);
    dim3 GridDim((Opts.Width + 16 - 1) / 16, (Opts.Height + 16 - 1) / 16);

    {
        vec2<f32> P0{0.5, 0.3};
        vec2<f32> P1{0.5, 0.5};

        DrawLine<<<GridDim, BlockDim>>>(d_Image, P0, P1, Opts.Width, Opts.Height);
    }
    // {
    //     vec2<f32> P0{0.3, 0.3};
    //     vec2<f32> P1{0.5, 0.6};

    //     DrawLine<<<GridDim, BlockDim>>>(d_Image, P0, P1, Opts.Width, Opts.Height);
    // }
    {
        vec2<f32> Circle{0.3f, 0.3f};
        f32 Radius = 0.05f;
        DrawCircle<<<GridDim, BlockDim>>>(d_Image, Circle, Radius, Opts.Width, Opts.Height);
    }

    u8 *Image = (u8 *)malloc(sizeof(u8) * Opts.Width * Opts.Height);
    cudaMemcpy(Image, d_Image, sizeof(u8) * Opts.Width * Opts.Height, cudaMemcpyDeviceToHost);

    stbi_write_jpg(Opts.OutputFilename, Opts.Width, Opts.Height, 1, Image, 100);

    free(Image);
    cudaFree(d_Image);
}
