#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define SOLUTION 3

#if SOLUTION == 1
// <1,1> wip version
__global__ void CategoricalCrossEntropyLoss(const float *logits, const int *true_labels, float *loss, int N, int C)
{
    *loss = 0.0f;
    for (int j = 0; j < N; j++)
    {
        float Loss_j = 0.0f;
        for (int k = 0; k < C; k++)
        {
            Loss_j += expf(logits[j * C + k]);
        }
        Loss_j = logf(Loss_j);
        Loss_j -= logits[j * C + true_labels[j]];
        *loss += Loss_j;
    }
    *loss /= N;
}
#elif SOLUTION == 2
// <N,1> wip version
// In this solution each thread calculates Loss_j
// And we use atomicAdd on loss with Loss_j / N
__global__ void CategoricalCrossEntropyLoss(const float *logits, const int *true_labels, float *loss, int N, int C)
{
    int j = blockIdx.x;
    float Loss_j = 0.0f;
    for (int k = 0; k < C; k++)
    {
        Loss_j += expf(logits[j * C + k]);
    }
    Loss_j = logf(Loss_j);
    Loss_j -= logits[j * C + true_labels[j]];
    atomicAdd(loss, Loss_j / N);
}
#elif SOLUTION == 3
// <N,C> version
// Each thread handles one element, block reduces to compute softmax denominator
__global__ void CategoricalCrossEntropyLoss(const float *logits, const int *true_labels, float *loss, int N, int C)
{
    int j = blockIdx.x;  // sample index
    int k = threadIdx.x; // class index
    
    __shared__ float shared_exp[1024]; // assuming C <= 1024
    
    // Each thread computes exp(logits[j*C + k])
    float exp_val = 0.0f;
    if (k < C) {
        exp_val = expf(logits[j * C + k]);
    }
    shared_exp[k] = exp_val;
    
    __syncthreads();
    
    // Block reduction to compute sum of exponentials
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (k % (2 * stride) == 0 && k + stride < C) {
            shared_exp[k] += shared_exp[k + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 computes the loss for this sample
    if (k == 0) {
        float Loss_j = logf(shared_exp[0]) - logits[j * C + true_labels[j]];
        atomicAdd(loss, Loss_j / N);
    }
}
#endif

// logits, true_labels, loss are device pointers
void solve(const float *logits, const int *true_labels, float *loss, int N, int C)
{
    printf("Using solution %d\n", SOLUTION);

#if SOLUTION == 1
    CategoricalCrossEntropyLoss<<<1, 1>>>(logits, true_labels, loss, N, C);
#elif SOLUTION == 2
    CategoricalCrossEntropyLoss<<<N, 1>>>(logits, true_labels, loss, N, C);
#elif SOLUTION == 3
    CategoricalCrossEntropyLoss<<<N, C>>>(logits, true_labels, loss, N, C);
#endif
}