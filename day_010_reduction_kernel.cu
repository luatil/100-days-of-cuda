/*
 * Day 010: Simple reduction
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

typedef float f32;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int32_t s32;

/*
 * Input: Vector with N elements
 * Output:
 * N: The length of the Input Vector
 *
 *  Concept: Tree-based reduction where threads work together
 *  Array: [1, 2, 3, 4, 5, 6, 7, 8] (8 threads)

 *  Step 1: Each pair adds together
 *  Thread 0: 1 + 2 = 3
 *  Thread 1: 3 + 4 = 7
 *  Thread 2: 5 + 6 = 11
 *  Thread 3: 7 + 8 = 15
 *  Result: [3, 7, 11, 15, _, _, _, _]

 *  Step 2: Every other thread works
 *  Thread 0: 3 + 7 = 10
 *  Thread 1: 11 + 15 = 26
 *  Result: [10, 26, _, _, _, _, _, _]

 *  Step 3: Final reduction
 *  Thread 0: 10 + 26 = 36
 *  Result: [36, _, _, _, _, _, _, _]
 *
 */

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

__global__ void ReduceVector(const f32 *Input, f32 *Output, int n)
{
    __shared__ f32 SharedData[BLOCK_SIZE];

    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    int Tx = threadIdx.x;

    // Load data into shared memory
    SharedData[Tx] = (Tid < n) ? Input[Tid] : 0.0f;
    __syncthreads();

    // Tree reduction - improved version without divergence
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (Tx < s)
        {
            SharedData[Tx] += SharedData[Tx + s];
        }
        __syncthreads();
    }

    // Thread 0 writes result for this block
    if (Tx == 0)
    {
        Output[blockIdx.x] = SharedData[0];
    }
}
