#ifndef CIRCLE_SDF
#define CIRCLE_SDF

#include "day_051_vector_2.cu"

template <typename T> static __device__ __host__ T CircleSDF(vec2<T> P, vec2<T> C, T R)
{
    return Distance(P, C) - R;
}

__device__ __host__ f32 LineSegmentSDF(vec2<f32> P, vec2<f32> A, vec2<f32> B, f32 R)
{
    float H = min(1.0, max(0.0, Dot(P - A, B - A) / Dot(B - A, B - A)));

    return Length(P - A - (B - A) * H) - R;
}

template <typename T> static __device__ __host__ T Abs(T X)
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

#endif
