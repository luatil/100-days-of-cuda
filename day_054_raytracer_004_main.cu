#include <stdlib.h>
#include <time.h>

#include "day_003_vendor_libraries.h"
#include "day_051_cli.h"
#include "day_051_types.h"
#include "day_051_vector_2.cu"

enum primitive_type
{
    PRIMITIVE_CIRCLE,
    PRIMITIVE_LINE
};

struct material
{
    u8 Color;
    f32 EdgeWidth;
};

struct primitive
{
    primitive_type Type;
    material Mat;
    union {
        struct
        {
            vec2<f32> Center;
            f32 Radius;
        } Circle;
        struct
        {
            vec2<f32> P0;
            vec2<f32> P1;
            f32 Width;
        } Line;
    };
};

struct scene
{
    primitive *Primitives;
    u32 Count;
};

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

        f32 Background = (f32)Image[Row * Width + Col];
        f32 Blended = Alpha * (f32)255 + (1.0f - Alpha) * Background;
        Image[Row * Width + Col] = (u8)Blended;
    }
}

__global__ void DrawLine(u8 *Image, vec2<f32> P0, vec2<f32> P1, u8 LineColor, u32 Width, u32 Height)
{
    s32 Row = blockDim.y * blockIdx.y + threadIdx.y;
    s32 Col = blockDim.x * blockIdx.x + threadIdx.x;

    if (Row < Height && Col < Width)
    {
        vec2<f32> Q = Screen2Point(Row, Col, Width, Height);
        f32 D = LineSegmentSDF(Q, P0, P1, 0.0005f);
        f32 Alpha = 1.0f - SmoothStep(0.0f, 0.002f, D);
        f32 Background = (f32)Image[Row * Width + Col];
        f32 Blended = Alpha * (f32)LineColor + (1.0f - Alpha) * Background;
        Image[Row * Width + Col] = (u8)Blended;
    }
}

__global__ void RenderScene(u8 *Image, scene SceneData, u32 Width, u32 Height)
{
    s32 Row = blockDim.y * blockIdx.y + threadIdx.y;
    s32 Col = blockDim.x * blockIdx.x + threadIdx.x;

    if (Row < Height && Col < Width)
    {
        vec2<f32> Q = Screen2Point(Row, Col, Width, Height);
        f32 Background = (f32)Image[Row * Width + Col];
        f32 FinalColor = Background;

        for (u32 I = 0; I < SceneData.Count; I++)
        {
            primitive Prim = SceneData.Primitives[I];
            f32 Alpha = 0.0f;

            if (Prim.Type == PRIMITIVE_CIRCLE)
            {
                f32 Val = CircleSDF(Q, Prim.Circle.Center, Prim.Circle.Radius);
                Alpha = 1.0f - SmoothStep(-Prim.Mat.EdgeWidth, Prim.Mat.EdgeWidth, Val);
            }
            else if (Prim.Type == PRIMITIVE_LINE)
            {
                f32 D = LineSegmentSDF(Q, Prim.Line.P0, Prim.Line.P1, Prim.Line.Width);
                Alpha = 1.0f - SmoothStep(0.0f, Prim.Mat.EdgeWidth, D);
            }

            FinalColor = Alpha * (f32)Prim.Mat.Color + (1.0f - Alpha) * FinalColor;
        }

        Image[Row * Width + Col] = (u8)FinalColor;
    }
}

primitive CreateCircle(vec2<f32> Center, f32 Radius, u8 Color = 255, f32 EdgeWidth = 0.001f)
{
    primitive Circle{};
    Circle.Type = PRIMITIVE_CIRCLE;
    Circle.Mat = {Color, EdgeWidth};
    Circle.Circle.Center = Center;
    Circle.Circle.Radius = Radius;
    return Circle;
}

primitive CreateLine(vec2<f32> P0, vec2<f32> P1, u8 Color = 255, f32 EdgeWidth = 0.002f, f32 Width = 0.0005f)
{
    primitive Line{};
    Line.Type = PRIMITIVE_LINE;
    Line.Mat = {Color, EdgeWidth};
    Line.Line.P0 = P0;
    Line.Line.P1 = P1;
    Line.Line.Width = Width;
    return Line;
}

f32 RandomFloat(f32 Min, f32 Max)
{
    return Min + (f32)rand() / RAND_MAX * (Max - Min);
}

u8 RandomColor()
{
    return (u8)(100 + rand() % 156);
}

void GenerateRandomCircles(primitive *Primitives, u32 &Count, u32 MaxPrimitives, u32 NumCircles)
{
    if (NumCircles == 0 || Count >= MaxPrimitives)
        return;

    vec2<f32> *CircleCenters = (vec2<f32> *)malloc(sizeof(vec2<f32>) * NumCircles);

    for (u32 I = 0; I < NumCircles && Count < MaxPrimitives; I++)
    {
        vec2<f32> Center = {RandomFloat(0.1f, 0.9f), RandomFloat(0.1f, 0.9f)};
        f32 Radius = RandomFloat(0.02f, 0.05f);
        u8 Color = RandomColor();

        CircleCenters[I] = Center;
        Primitives[Count++] = CreateCircle(Center, Radius, Color);
    }

    u32 NumConnections = NumCircles + rand() % NumCircles;
    for (u32 I = 0; I < NumConnections && Count < MaxPrimitives; I++)
    {
        u32 Index1 = rand() % NumCircles;
        u32 Index2 = rand() % NumCircles;

        if (Index1 != Index2)
        {
            u8 LineColor = RandomColor();
            Primitives[Count++] = CreateLine(CircleCenters[Index1], CircleCenters[Index2], LineColor);
        }
    }

    free(CircleCenters);
}

void GenerateConnectedCircles(primitive *Primitives, u32 &Count, u32 MaxPrimitives, u32 NumCircles)
{
    if (NumCircles == 0 || Count >= MaxPrimitives)
        return;

    vec2<f32> *CircleCenters = (vec2<f32> *)malloc(sizeof(vec2<f32>) * NumCircles);

    for (u32 I = 0; I < NumCircles && Count < MaxPrimitives; I++)
    {
        f32 Angle = (f32)I / (f32)NumCircles * 2.0f * 3.14159f;
        f32 Radius = 0.3f;
        vec2<f32> Center = {0.5f + Radius * cosf(Angle), 0.5f + Radius * sinf(Angle)};
        CircleCenters[I] = Center;

        Primitives[Count++] = CreateCircle(Center, 0.03f, 200);
    }

    for (u32 I = 0; I < NumCircles && Count < MaxPrimitives; I++)
    {
        u32 NextIndex = (I + 1) % NumCircles;
        Primitives[Count++] = CreateLine(CircleCenters[I], CircleCenters[NextIndex], 150);
    }

    for (u32 I = 0; I < NumCircles && Count < MaxPrimitives; I++)
    {
        vec2<f32> Center = {0.5f, 0.5f};
        Primitives[Count++] = CreateLine(CircleCenters[I], Center, 100);
    }

    if (Count < MaxPrimitives)
    {
        Primitives[Count++] = CreateCircle({0.5f, 0.5f}, 0.02f, 255);
    }

    free(CircleCenters);
}

int main(int argc, char *argv[])
{
    options Opts = ParseCommandLine(argc, argv);

    srand((unsigned int)time(NULL));

    u8 *DImage;
    cudaMalloc(&DImage, sizeof(u8) * Opts.Width * Opts.Height);
    cudaMemset(DImage, 0, sizeof(u8) * Opts.Width * Opts.Height);

    u32 MaxPrimitives = 1024;
    primitive *Primitives = (primitive *)malloc(sizeof(primitive) * MaxPrimitives);
    u32 PrimitiveCount = 0;

    u32 NumCircles = 64;
    GenerateRandomCircles(Primitives, PrimitiveCount, MaxPrimitives, NumCircles);

    primitive *DPrimitives;
    cudaMalloc(&DPrimitives, sizeof(primitive) * PrimitiveCount);
    cudaMemcpy(DPrimitives, Primitives, sizeof(primitive) * PrimitiveCount, cudaMemcpyHostToDevice);

    scene SceneData = {DPrimitives, PrimitiveCount};

    dim3 BlockDim(16, 16);
    dim3 GridDim((Opts.Width + 16 - 1) / 16, (Opts.Height + 16 - 1) / 16);

    RenderScene<<<GridDim, BlockDim>>>(DImage, SceneData, Opts.Width, Opts.Height);

    u8 *Image = (u8 *)malloc(sizeof(u8) * Opts.Width * Opts.Height);
    cudaMemcpy(Image, DImage, sizeof(u8) * Opts.Width * Opts.Height, cudaMemcpyDeviceToHost);

    stbi_write_jpg(Opts.OutputFilename, Opts.Width, Opts.Height, 1, Image, 100);

    free(Image);
    free(Primitives);
    cudaFree(DImage);
    cudaFree(DPrimitives);
}
