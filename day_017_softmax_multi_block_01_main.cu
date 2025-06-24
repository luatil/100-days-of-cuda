#include "day_015_common.h"
#include <cfloat>
#include <math.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// Phase 1: Each block finds its local maximum
__global__ void softmax_find_max_kernel(float *input, float *global_max_per_block, int seq_len, int elements_per_block)
{
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    int start_idx = block_id * elements_per_block;
    int end_idx = min(start_idx + elements_per_block, seq_len);

    __shared__ float sdata[BLOCK_SIZE];

    // Find local maximum for this block's chunk
    float thread_max = -FLT_MAX;
    for (int i = start_idx + tid; i < end_idx; i += blockDim.x)
    {
        thread_max = fmaxf(thread_max, input[i]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    // Block-level reduction to find local max
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Store this block's maximum
    if (tid == 0)
    {
        global_max_per_block[block_id] = sdata[0];
    }
}

// Phase 2: Reduce all local maxes to find global max
__global__ void softmax_reduce_max_kernel(float *global_max_per_block, float *global_max, int num_blocks)
{
    int tid = threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE];

    float thread_max = -FLT_MAX;
    for (int i = tid; i < num_blocks; i += blockDim.x)
    {
        thread_max = fmaxf(thread_max, global_max_per_block[i]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        *global_max = sdata[0];
    }
}

// Phase 3: Each block computes local sum of exp(x - global_max)
__global__ void softmax_compute_sum_kernel(float *input, float *global_sum_per_block, float global_max, int seq_len,
                                           int elements_per_block)
{
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    int start_idx = block_id * elements_per_block;
    int end_idx = min(start_idx + elements_per_block, seq_len);

    __shared__ float sdata[BLOCK_SIZE];

    // Compute local sum of exponentials
    float thread_sum = 0.0f;
    for (int i = start_idx + tid; i < end_idx; i += blockDim.x)
    {
        thread_sum += expf(input[i] - global_max);
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction to find local sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Store this block's sum
    if (tid == 0)
    {
        global_sum_per_block[block_id] = sdata[0];
    }
}

// Phase 4: Reduce all local sums to find global sum
__global__ void softmax_reduce_sum_kernel(float *global_sum_per_block, float *global_sum, int num_blocks)
{
    int tid = threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE];

    float thread_sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x)
    {
        thread_sum += global_sum_per_block[i];
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        *global_sum = sdata[0];
    }
}

// Phase 5: Compute final softmax values
__global__ void softmax_finalize_kernel(float *input, float *output, float global_max, float global_sum, int seq_len,
                                        int elements_per_block)
{
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    int start_idx = block_id * elements_per_block;
    int end_idx = min(start_idx + elements_per_block, seq_len);

    for (int i = start_idx + tid; i < end_idx; i += blockDim.x)
    {
        output[i] = expf(input[i] - global_max) / global_sum;
    }
}

void softmax_multi_block(float *input, float *output, int seq_len)
{
    int elements_per_block = BLOCK_SIZE * 4; // Each block handles 1024 elements
    int num_blocks = (seq_len + elements_per_block - 1) / elements_per_block;

    // Allocate intermediate arrays
    float *global_max_per_block, *global_sum_per_block;
    float *global_max, *global_sum;

    cudaMalloc(&global_max_per_block, sizeof(float) * num_blocks);
    cudaMalloc(&global_sum_per_block, sizeof(float) * num_blocks);
    cudaMalloc(&global_max, sizeof(float));
    cudaMalloc(&global_sum, sizeof(float));

    // Phase 1: Find local maxes
    softmax_find_max_kernel<<<num_blocks, BLOCK_SIZE>>>(input, global_max_per_block, seq_len, elements_per_block);
    cudaDeviceSynchronize();

    // Phase 2: Reduce to global max
    softmax_reduce_max_kernel<<<1, BLOCK_SIZE>>>(global_max_per_block, global_max, num_blocks);
    cudaDeviceSynchronize();

    // Phase 3: Compute local sums
    softmax_compute_sum_kernel<<<num_blocks, BLOCK_SIZE>>>(input, global_sum_per_block, *global_max, seq_len,
                                                           elements_per_block);
    cudaDeviceSynchronize();

    // Phase 4: Reduce to global sum
    softmax_reduce_sum_kernel<<<1, BLOCK_SIZE>>>(global_sum_per_block, global_sum, num_blocks);
    cudaDeviceSynchronize();

    // Phase 5: Compute final softmax
    softmax_finalize_kernel<<<num_blocks, BLOCK_SIZE>>>(input, output, *global_max, *global_sum, seq_len,
                                                        elements_per_block);

    // Cleanup
    cudaFree(global_max_per_block);
    cudaFree(global_sum_per_block);
    cudaFree(global_max);
    cudaFree(global_sum);
}

int main()
{
    const u32 seq_len = 10000; // Much larger than 1024!
    const u32 N = seq_len;

    f32 *Input = AllocateCPU(f32, N);
    f32 *Output = AllocateCPU(f32, N);

    // Initialize with test values
    for (u32 I = 0; I < N; I++)
    {
        Input[I] = (f32)(I % 10) + 1.0f;
    }

    f32 *Device_Input, *Device_Output;
    cudaMalloc(&Device_Input, sizeof(f32) * N);
    cudaMalloc(&Device_Output, sizeof(f32) * N);

    cudaMemcpy(Device_Input, Input, sizeof(f32) * N, cudaMemcpyHostToDevice);

    softmax_multi_block(Device_Input, Device_Output, seq_len);

    cudaMemcpy(Output, Device_Output, sizeof(f32) * N, cudaMemcpyDeviceToHost);

    // Verify softmax properties
    f32 sum = 0.0f;
    for (u32 I = 0; I < seq_len; I++)
    {
        sum += Output[I];
    }

    fprintf(stdout, "Sequence length: %d\n", seq_len);
    fprintf(stdout, "Softmax sum: %f (should be ~1.0)\n", sum);
    fprintf(stdout, "First 5 values: ");
    for (u32 I = 0; I < 5; I++)
    {
        fprintf(stdout, "%.6f ", Output[I]);
    }
    fprintf(stdout, "\n");

    Assert(fabsf(sum - 1.0f) < 0.001f);

    FreeCPU(Input);
    FreeCPU(Output);
    cudaFree(Device_Input);
    cudaFree(Device_Output);
}
