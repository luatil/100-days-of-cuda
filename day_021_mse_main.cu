/*
 * Calculates MSE
 *
 * Input:  predictions = [1.0, 2.0, 3.0, 4.0]
 * targets = [1.5, 2.5, 3.5, 4.5]
 * Output: mse = 0.25
 */

#include <stdio.h>

#ifndef SOLUTION
#define SOLUTION 3
#endif

#if SOLUTION == 1
__device__ float Square(float x)
{
    return x * x;
}
__global__ void mse_kernel(const float *predictions, const float *targets, float *mse, int N)
{
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
    {
        sum += Square(targets[i] - predictions[i]);
    }
    *mse = sum / N;
}

// predictions, targets, mse are device pointers
void solve(const float *predictions, const float *targets, float *mse, int N)
{
    mse_kernel<<<1, 1>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();
}
#elif SOLUTION == 2
#define BLOCK_DIM 256
#define COARSE_FACTOR 4

__device__ float Square(float X)
{
    return X * X;
}

__global__ void MseKernel(const float *Pred, const float *Target, float *Mse, int N)
{
    __shared__ float SharedMem[BLOCK_DIM];

    int Tid = COARSE_FACTOR * BLOCK_DIM * blockIdx.x + threadIdx.x;
    int Tx = threadIdx.x;

    float Sum = 0.0f;
    for (int I = 0; I < COARSE_FACTOR; I++)
    {
        int Pos = Tid + BLOCK_DIM * I;
        if (Pos < N)
        {
            Sum += Square(Pred[Pos] - Target[Pos]) / N;
        }
    }
    SharedMem[Tx] = Sum;

    for (int Stride = BLOCK_DIM / 2; Stride > 0; Stride >>= 1)
    {
        __syncthreads();
        if (Tx < Stride)
        {
            SharedMem[Tx] += SharedMem[Tx + Stride];
        }
    }

    __syncthreads();
    if (Tx == 0)
    {
        atomicAdd(Mse, SharedMem[0]);
    }
}

void solve(const float *predictions, const float *targets, float *mse, int N)
{
    int GridDim = (N + (BLOCK_DIM * COARSE_FACTOR) - 1) / (BLOCK_DIM * COARSE_FACTOR);
    MseKernel<<<GridDim, BLOCK_DIM>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();
}
#elif SOLUTION == 3
#define BLOCK_DIM (256)
#define COARSE_FACTOR 4
#define WARP_SIZE 32

__device__ float Square(float X)
{
    return X * X;
}

__device__ __forceinline__ float warpReduceSum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val)
{
    static __shared__ float shared[WARP_SIZE]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val); // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads(); // Wait for all partial reductions

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceSum(val); // Final reduce within first warp

    return val;
}

__global__ void MseKernel(const float *Pred, const float *Target, float *Mse, int N)
{
    int Tid = COARSE_FACTOR * BLOCK_DIM * blockIdx.x + threadIdx.x;

    float Sum = 0.0f;
    for (int I = 0; I < COARSE_FACTOR; I++)
    {
        int Pos = Tid + BLOCK_DIM * I;
        if (Pos < N)
        {
            Sum += Square(Pred[Pos] - Target[Pos]) / N;
        }
    }

    Sum = blockReduceSum(Sum);

    if (threadIdx.x == 0)
    {
        atomicAdd(Mse, Sum);
    }
}

void solve(const float *predictions, const float *targets, float *mse, int N)
{
    int GridDim = (N + (BLOCK_DIM * COARSE_FACTOR) - 1) / (BLOCK_DIM * COARSE_FACTOR);
    MseKernel<<<GridDim, BLOCK_DIM>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();
}
#elif SOLUTION == 4
#else
#endif

int main()
{
    float pred[] = {1.0, 2.0f, 3.0f, 4.0f};
    float tgt[] = {1.5, 2.5, 3.5, 4.5};
    float eo = 0.25;

    int sizeb = sizeof(pred);

    float *d_pred, *d_tgt, *d_mse;

    cudaMalloc(&d_pred, sizeb);
    cudaMalloc(&d_tgt, sizeb);
    cudaMalloc(&d_mse, sizeof(float));

    cudaMemcpy(d_pred, pred, sizeb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt, tgt, sizeb, cudaMemcpyHostToDevice);

    solve(d_pred, d_tgt, d_mse, sizeof(pred) / sizeof(pred[0]));

    float mse = 0.0f;
    cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);

    printf("mse = %.3f\n", mse);
    printf("eo = %.3f\n", eo);
}
