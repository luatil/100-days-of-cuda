#include <stdlib.h>
#include <time.h>

#include "day_003_vendor_libraries.h"
#include "day_051_cli.h"
#include "day_051_types.h"
#include "day_051_vector_2.cu"
#include "day_055_random.cu"
#include "day_055_sdf.cu"

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

struct ray
{
    vec2<f32> Origin;
    vec2<f32> Direction;
};

struct image
{
    u8 *m_Data;
    u32 m_Width;
    u32 m_Height;

    image(u32 Width, u32 Height)
    {
        m_Width = Width;
        m_Height = Height;
        cudaMalloc(&this->m_Data, Width * Height * sizeof(u8));
        this->Clear();
    }

    void Clear()
    {
        cudaMemset(this->m_Data, 0, this->m_Width * this->m_Height * sizeof(u8));
    }

    void Save(const char *Filename)
    {
        u8 *Image = (u8 *)malloc(sizeof(u8) * m_Width * m_Height);
        cudaMemcpy(Image, m_Data, sizeof(u8) * m_Width * m_Height, cudaMemcpyDeviceToHost);
        stbi_write_jpg(Filename, m_Width, m_Height, 1, Image, 100);
        free(Image);
    }

    ~image()
    {
        cudaFree(this->m_Data);
    }
};

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

        for (u32 i = 0; i < SceneData.Count; i++)
        {
            primitive Prim = SceneData.Primitives[i];
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

static primitive CreateCircle(vec2<f32> Center, f32 Radius, u8 Color = 255, f32 EdgeWidth = 0.001f)
{
    primitive Circle{};
    Circle.Type = PRIMITIVE_CIRCLE;
    Circle.Mat = {Color, EdgeWidth};
    Circle.Circle.Center = Center;
    Circle.Circle.Radius = Radius;
    return Circle;
}

static primitive CreateLine(vec2<f32> P0, vec2<f32> P1, u8 Color = 255, f32 EdgeWidth = 0.002f, f32 Width = 0.0005f)
{
    primitive Line{};
    Line.Type = PRIMITIVE_LINE;
    Line.Mat = {Color, EdgeWidth};
    Line.Line.P0 = P0;
    Line.Line.P1 = P1;
    Line.Line.Width = Width;
    return Line;
}

static void GenerateRandomCircles(primitive *Primitives, u32 &Count, u32 MaxPrimitives, u32 NumCircles)
{
    if (NumCircles == 0 || Count >= MaxPrimitives)
        return;

    vec2<f32> *CircleCenters = (vec2<f32> *)malloc(sizeof(vec2<f32>) * NumCircles);

    for (u32 i = 0; i < NumCircles && Count < MaxPrimitives; i++)
    {
        vec2<f32> Center = {RandomFloat(0.1f, 0.9f), RandomFloat(0.1f, 0.9f)};
        f32 Radius = RandomFloat(0.02f, 0.05f);
        u8 Color = RandomColor();

        CircleCenters[i] = Center;
        Primitives[Count++] = CreateCircle(Center, Radius, Color);
    }

    u32 NumConnections = NumCircles + rand() % NumCircles;
    for (u32 i = 0; i < NumConnections && Count < MaxPrimitives; i++)
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


int main(int argc, char *argv[])
{
    options Opts = ParseCommandLine(argc, argv);

    srand((unsigned int)time(NULL));

    image Image(Opts.Width, Opts.Height);

    u32 MaxPrimitives = 1024;
    primitive *Primitives = (primitive *)malloc(sizeof(primitive) * MaxPrimitives);
    u32 PrimitiveCount = 0;

    u32 NumCircles = 64;
    GenerateRandomCircles(Primitives, PrimitiveCount, MaxPrimitives, NumCircles);

    primitive *d_Primitives;
    cudaMalloc(&d_Primitives, sizeof(primitive) * PrimitiveCount);
    cudaMemcpy(d_Primitives, Primitives, sizeof(primitive) * PrimitiveCount, cudaMemcpyHostToDevice);

    scene SceneData = {d_Primitives, PrimitiveCount};

    dim3 BlockDim(16, 16);
    dim3 GridDim((Opts.Width + 16 - 1) / 16, (Opts.Height + 16 - 1) / 16);

    RenderScene<<<GridDim, BlockDim>>>(Image.m_Data, SceneData, Opts.Width, Opts.Height);

    Image.Save("temp.jpg");

    free(Primitives);
    cudaFree(d_Primitives);
}
