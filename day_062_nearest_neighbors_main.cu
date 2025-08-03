#include <cuda_runtime.h>
#include <cfloat>
#include <stdio.h>
// #define LEET_GPU

// Ideas to improve this:
// In some way or another we need to use some sort of memory technique
// to speed up the process.
//
// One of them could be to use a grid and only look at elements that fall
// occur withing that grid.
//
// But this makes assumptions on what the grid is actually like.

__global__ void NearestNeighbors(const float *Points, int *Indices, int N)
{
    int PointIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (PointIdx < N)
    {
        // Iterate over all of the other points
        // Check which one is closer to our current point
        float MinDistanceSquared = FLT_MAX;
        for (int I = 0; I < N; I++)
        {
            if (PointIdx != I)
            {
                float Distance = powf(Points[3 * PointIdx] - Points[3 * I], 2) +
                                 powf(Points[3 * PointIdx + 1] - Points[3 * I + 1], 2) +
                                 powf(Points[3 * PointIdx + 2] - Points[3 * I + 2], 2);
                if (Distance < MinDistanceSquared)
                {
                    Indices[PointIdx] = I;
                    MinDistanceSquared = Distance;
                }
            }
        }
    }
}

extern "C" void solve(const float* points, int* indices, int N)
{
    const int BlockDim = 256;
    const int GridDim = (N + BlockDim - 1) / BlockDim;
    NearestNeighbors<<<GridDim, BlockDim>>>(points, indices, N);
}

#ifndef LEET_GPU
int main()
{
    // points  = [(0,0,0), (1,0,0), (5,5,5)]
    //         indices = [-1, -1, -1]
    //         N       = 3
    float Points[] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 5.0f, 5.0f};
    int Indices[] = {-1, -1, -1};
    int N = 3;

    float *d_Points;
    int  *d_Indices;
    cudaMalloc(&d_Points, N * 3 * sizeof(float));
    cudaMalloc(&d_Indices, N * sizeof(int));

    cudaMemcpy(d_Points, Points, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Indices, Indices, N * sizeof(int), cudaMemcpyHostToDevice);

    solve(d_Points, d_Indices, N);

    cudaMemcpy(Indices, d_Indices, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf("%d => %d\n", i, Indices[i]);
    }
}
#endif
