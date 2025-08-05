#include <stdlib.h>
#include <time.h>

#include "day_003_vendor_libraries.h"
#include "day_051_cli.h"
#include "day_051_types.h"
#include "day_051_vector_2.cu"
#include "day_055_random.cu"
#include "day_055_sdf.cu"

const f32 PI = 3.14159265358979323846;

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
    u8 *MData;
    u32 MWidth;
    u32 MHeight;

    image(u32 Width, u32 Height)
    {
        MWidth = Width;
        MHeight = Height;
        cudaMalloc(&this->MData, Width * Height * sizeof(u8));
        this->Clear();
    }

    void Clear()
    {
        cudaMemset(this->MData, 0, this->MWidth * this->MHeight * sizeof(u8));
    }

    void Save(const char *Filename)
    {
        u8 *Image = (u8 *)malloc(sizeof(u8) * MWidth * MHeight);
        cudaMemcpy(Image, MData, sizeof(u8) * MWidth * MHeight, cudaMemcpyDeviceToHost);
        stbi_write_jpg(Filename, MWidth, MHeight, 1, Image, 100);
        // stbi_write_png(Filename, m_Width, m_Height, 1, Image, m_Width);
        free(Image);
    }

    ~image()
    {
        cudaFree(this->MData);
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

    for (u32 I = 0; I < NumCircles && Count < MaxPrimitives; I++)
    {
        vec2<f32> Center = {RandomFloat(0.1f, 0.9f), RandomFloat(0.1f, 0.9f)};
        f32 Radius = RandomFloat(0.02f, 0.03f);
        u8 Color = RandomColor(10, 250);

        CircleCenters[I] = Center;
        Primitives[Count++] = CreateCircle(Center, Radius, Color);
    }

    free(CircleCenters);
}

// Ray to line
primitive LineFromRay(ray Ray, u8 Color = 100, f32 EdgeWidth = 0.005f)
{
    primitive Line = CreateLine(Ray.Origin, Ray.Origin + Ray.Direction, Color, EdgeWidth);
    return Line;
}

int main(int argc, char *argv[])
{
    options Opts = ParseCommandLine(argc, argv);

    srand((unsigned int)time(NULL));

    image Image(Opts.Width, Opts.Height);

    u32 MaxPrimitives = 1024 * 1024;
    primitive *Primitives = (primitive *)malloc(sizeof(primitive) * MaxPrimitives);
    u32 PrimitiveCount = 0;

    u32 NumCircles = 1;
    // GenerateRandomCircles(Primitives, PrimitiveCount, MaxPrimitives, NumCircles);

    primitive Circle1 = CreateCircle({0.3f, 0.4f}, 0.01f, 255);
    PrimitiveCount++;
    Primitives[0] = Circle1;

    u32 NumberOfRays = 256;
    for (u32 I = 0; I < NumberOfRays; I++)
    {
        f32 Angle = (2 * PI / NumberOfRays) * I;
        f32 X = cosf(Angle);
        f32 Y = sinf(Angle);
        ray Ray{Circle1.Circle.Center, {X, Y}};
        // Put a Line from a ray
        if (I % 1 == 0)
        {
            primitive Line = LineFromRay(Ray, 100, 0.001f);
            Primitives[PrimitiveCount++] = Line;
        }
    }

    primitive *DPrimitives;
    cudaMalloc(&DPrimitives, sizeof(primitive) * PrimitiveCount);
    cudaMemcpy(DPrimitives, Primitives, sizeof(primitive) * PrimitiveCount, cudaMemcpyHostToDevice);

    scene SceneData = {DPrimitives, PrimitiveCount};

    dim3 BlockDim(16, 16);
    dim3 GridDim((Opts.Width + 16 - 1) / 16, (Opts.Height + 16 - 1) / 16);

    RenderScene<<<GridDim, BlockDim>>>(Image.MData, SceneData, Opts.Width, Opts.Height);

    Image.Save("temp.jpg");

    free(Primitives);
    cudaFree(DPrimitives);
}
