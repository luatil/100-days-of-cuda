#include <cmath>
#include <stdio.h>

#include "day_003_vendor_libraries.h"

typedef unsigned int u32;
typedef unsigned char u8;
typedef float f32;

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

typedef int s32;
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

    __device__ __host__ T dot(const vec2 &Other) const
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

int main()
{
    s32 Width = 400;
    s32 Height = 800;

    u8 *d_Image;

    cudaMalloc(&d_Image, sizeof(u8) * Width * Height);

    dim3 BlockDim(16, 16);
    dim3 GridDim((Width + 16 - 1) / 16, (Height + 16 - 1) / 16);

    vec2<s32> Circle{200, 200};
    s32 Radius = 100;

    DrawCircle<<<GridDim, BlockDim>>>(d_Image, Circle, Radius, Width, Height);

    u8 *Image = (u8 *)malloc(sizeof(u8) * Width * Height);
    cudaMemcpy(Image, d_Image, sizeof(u8) * Width * Height, cudaMemcpyDeviceToHost);

    const char *OutFilename = "temp.jpg";
    stbi_write_jpg(OutFilename, Width, Height, 1, Image, 100);
}
