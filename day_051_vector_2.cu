#ifndef VEC2_H
#define VEC2_H

#include <cmath>
#include <getopt.h>
#include <stdlib.h>

#include "day_051_types.h"

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

template <typename T> __device__ __host__ vec2<T> Normalize(vec2<T> V)
{
    return V / V.Length();
}

template <typename T> __device__ __host__ T Dot(vec2<T> V, vec2<T> W)
{
    return V.X * W.X + V.Y * W.Y;
}

__device__ __host__ f32 Length(vec2<float> V)
{
    return SquareRoot(V.LengthSquared());
}

#endif
