/*
 * NAME
 * 	circlesdf - renders a circle using a signed distance function
 *
 * SYNOPSIS
 * 	circlesdf [OPTION]...
 *
 * DESCRIPTION
 * 	Renders a circle using CUDA and signed distance field (SDF) technique.
 * 	Outputs the result as a JPEG image with anti-aliased edges.
 *
 * 	-w, --width=WIDTH
 * 		set image width in pixels (default: 400)
 *
 * 	-h, --height=HEIGHT
 * 		set image height in pixels (default: 400)
 *
 * 	-o, --output-filename=FILE
 * 		output filename for the generated image (default: temp.jpg)
 *
 * 	-x, --cx=X
 * 		circle center X coordinate (default: 200)
 *
 * 	-y, --cy=Y
 * 		circle center Y coordinate (default: 200)
 *
 * 	-r, --radius=R
 * 		circle radius in pixels (default: 100)
 *
 * 	--help
 * 		display this help and exit
 *
 */
#include <cmath>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include "day_003_vendor_libraries.h"

typedef unsigned int u32;
typedef unsigned char u8;
typedef float f32;
typedef int s32;

typedef struct
{
    int Width;
    int Height;
    char *OutputFilename;
    int CircleX;
    int CircleY;
    int Radius;
} options;

template <typename T> T __device__ __host__ SquareRoot(T X)
{
#ifdef __CUDA_ARCH__
    return sqrtf(X);
#else
    return sqrt(X);
#endif
}

__device__ __host__ float Clamp(float X, float Lower = 0.0f, float Upper = 1.0f)
{
    if (X < Lower)
        return Lower;
    if (X > Upper)
        return Upper;
    return X;
}

__device__ __host__ f32 SmoothStep(f32 Edge0, f32 Edge1, f32 X)
{
    // Scale, and clamp x to 0..1 range
    X = Clamp((X - Edge0) / (Edge1 - Edge0), 0.0f, 1.0f);

    return X * X * (3.0f - 2.0f * X);
}
template <typename T> struct vec2
{
    T X;
    T Y;

    __device__ __host__ vec2() : X(0), Y(0)
    {
    }
    __device__ __host__ vec2(T X, T Y) : X(X), Y(Y)
    {
    }

    __device__ __host__ vec2 operator+(const vec2 &Other) const
    {
        return vec2(X + Other.X, Y + Other.Y);
    }

    __device__ __host__ vec2 operator-(const vec2 &Other) const
    {
        return vec2(X - Other.X, Y - Other.Y);
    }

    __device__ __host__ vec2 operator*(T Scalar) const
    {
        return vec2(X * Scalar, Y * Scalar);
    }

    __device__ __host__ vec2 operator/(T Scalar) const
    {
        return vec2(X / Scalar, Y / Scalar);
    }

    __device__ __host__ vec2 &operator+=(const vec2 &Other)
    {
        X += Other.X;
        Y += Other.Y;
        return *this;
    }

    __device__ __host__ vec2 &operator-=(const vec2 &Other)
    {
        X -= Other.X;
        Y -= Other.Y;
        return *this;
    }

    __device__ __host__ vec2 &operator*=(T Scalar)
    {
        X *= Scalar;
        Y *= Scalar;
        return *this;
    }

    __device__ __host__ vec2 &operator/=(T Scalar)
    {
        X /= Scalar;
        Y /= Scalar;
        return *this;
    }

    __device__ __host__ bool operator==(const vec2 &Other) const
    {
        return X == Other.X && Y == Other.Y;
    }

    __device__ __host__ bool operator!=(const vec2 &Other) const
    {
        return !(*this == Other);
    }

    __device__ __host__ T Dot(const vec2 &Other) const
    {
        return X * Other.X + Y * Other.Y;
    }

    __device__ __host__ T LengthSquared() const
    {
        return X * X + Y * Y;
    }

    __device__ __host__ T Length() const
    {
        return SquareRoot(LengthSquared());
    }
};

template <typename T> __device__ __host__ T Square(T X)
{
    return X * X;
}

template <typename T> __device__ __host__ T Distance(vec2<T> P, vec2<T> Q)
{
    return (P - Q).Length();
}

template <typename T> __device__ __host__ T CircleSDF(vec2<T> P, vec2<T> C, T R)
{
    return Distance(P, C) - R;
}

static void PrintUsage(const char *ProgramName)
{
    printf("Usage: %s [OPTION]...\n", ProgramName);
    printf("Renders a circle using CUDA and signed distance field (SDF) technique.\n");
    printf("Outputs the result as a JPEG image with anti-aliased edges.\n\n");
    printf("Options:\n");
    printf("  -w, --width=WIDTH          set image width in pixels (default: 400)\n");
    printf("  -h, --height=HEIGHT        set image height in pixels (default: 400)\n");
    printf("  -o, --output-filename=FILE output filename (default: temp.jpg)\n");
    printf("  -x, --cx=X                 circle center X coordinate (default: 200)\n");
    printf("  -y, --cy=Y                 circle center Y coordinate (default: 200)\n");
    printf("  -r, --radius=R             circle radius in pixels (default: 100)\n");
    printf("      --help                 display this help and exit\n");
}

static options ParseCommandLine(int ArgumentCount, char *Arguments[])
{
    options Opts = {400, 400, (char *)"temp.jpg", 200, 200, 100};

    static struct option LongOptions[] = {{"width", required_argument, 0, 'w'},
                                          {"height", required_argument, 0, 'h'},
                                          {"output-filename", required_argument, 0, 'o'},
                                          {"cx", required_argument, 0, 'x'},
                                          {"cy", required_argument, 0, 'y'},
                                          {"radius", required_argument, 0, 'r'},
                                          {"help", no_argument, 0, '?'},
                                          {0, 0, 0, 0}};

    int OptionIndex = 0;
    int Char;

    while ((Char = getopt_long(ArgumentCount, Arguments, "w:h:o:x:y:r:", LongOptions, &OptionIndex)) != -1)
    {
        switch (Char)
        {
        case 'w':
            Opts.Width = atoi(optarg);
            if (Opts.Width <= 0)
            {
                fprintf(stderr, "Error: Width must be positive\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'h':
            Opts.Height = atoi(optarg);
            if (Opts.Height <= 0)
            {
                fprintf(stderr, "Error: Height must be positive\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'o':
            Opts.OutputFilename = optarg;
            break;
        case 'x':
            Opts.CircleX = atoi(optarg);
            break;
        case 'y':
            Opts.CircleY = atoi(optarg);
            break;
        case 'r':
            Opts.Radius = atoi(optarg);
            if (Opts.Radius <= 0)
            {
                fprintf(stderr, "Error: Radius must be positive\n");
                exit(EXIT_FAILURE);
            }
            break;
        case '?':
        default:
            PrintUsage(Arguments[0]);
            exit(EXIT_SUCCESS);
            break;
        }
    }

    return Opts;
}

__global__ void DrawCircle(u8 *Image, vec2<s32> Circle, s32 Radius, u32 Width, u32 Height)
{
    s32 Row = blockDim.y * blockIdx.y + threadIdx.y;
    s32 Col = blockDim.x * blockIdx.x + threadIdx.x;

    if (Row < Height && Col < Width)
    {
        vec2<s32> P(Row, Col);
        s32 Val = CircleSDF(P, Circle, Radius);
        float EdgeWidth = 5.0f;
        float Alpha = 1.0f - SmoothStep(-EdgeWidth, EdgeWidth, (float)Val);
        Image[Row * Width + Col] = (u8)(Alpha * 255);
    }
}

int main(int argc, char *argv[])
{
    options Opts = ParseCommandLine(argc, argv);

    u8 *DImage;

    cudaMalloc(&DImage, sizeof(u8) * Opts.Width * Opts.Height);

    dim3 BlockDim(16, 16);
    dim3 GridDim((Opts.Width + 16 - 1) / 16, (Opts.Height + 16 - 1) / 16);

    vec2<s32> Circle{Opts.CircleX, Opts.CircleY};

    DrawCircle<<<GridDim, BlockDim>>>(DImage, Circle, Opts.Radius, Opts.Width, Opts.Height);

    u8 *Image = (u8 *)malloc(sizeof(u8) * Opts.Width * Opts.Height);
    cudaMemcpy(Image, DImage, sizeof(u8) * Opts.Width * Opts.Height, cudaMemcpyDeviceToHost);

    stbi_write_jpg(Opts.OutputFilename, Opts.Width, Opts.Height, 1, Image, 100);

    free(Image);
    cudaFree(DImage);
}
