#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Helper function to get the next power of 2
__device__ __host__ int nextPowerOf2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Parallel reduction to find max element and its index
__global__ void findMaxKernel(float *input, int N, float *maxVal, int *maxIdx)
{
    extern __shared__ float sdata[];
    float *smax = sdata;
    int *sidx = (int *)&smax[blockDim.x];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load data
    if (i < N)
    {
        smax[tid] = input[i];
        sidx[tid] = i;
    }
    else
    {
        smax[tid] = -FLT_MAX;
        sidx[tid] = -1;
    }
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && i + s < N)
        {
            if (smax[tid] < smax[tid + s])
            {
                smax[tid] = smax[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0)
    {
        atomicMax((int *)maxVal, __float_as_int(smax[0]));
        if (__int_as_float(*(int *)maxVal) == smax[0])
        {
            *maxIdx = sidx[0];
        }
    }
}

// Bitonic sort for small k (works well for k <= 1024)
__global__ void bitonicSortKernel(float *data, int n, int j, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = i ^ j;

    if (ixj > i && i < n && ixj < n)
    {
        if ((i & k) == 0)
        {
            // Sort ascending in this part
            if (data[i] < data[ixj])
            {
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
        else
        {
            // Sort descending in this part
            if (data[i] > data[ixj])
            {
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// Parallel partial sort using bitonic sort
__global__ void partialBitonicSort(float *input, float *output, int N, int k)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    // Each block handles a portion and finds local top-k
    int elemsPerBlock = (N + gridDim.x - 1) / gridDim.x;
    int start = blockIdx.x * elemsPerBlock;
    int end = min(start + elemsPerBlock, N);
    int localN = end - start;

    // Load k elements (or all if less than k)
    int loadCount = min(k, localN);
    if (tid < loadCount && start + tid < N)
    {
        shared[tid] = input[start + tid];
    }
    else if (tid < k)
    {
        shared[tid] = -FLT_MAX;
    }
    __syncthreads();

    // Bitonic sort the k elements
    int sortSize = nextPowerOf2(k);
    for (int size = 2; size <= sortSize; size <<= 1)
    {
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            if (tid < k)
            {
                int partner = tid ^ stride;
                if (partner < k)
                {
                    bool ascending = ((tid & size) == 0);
                    float val1 = shared[tid];
                    float val2 = shared[partner];

                    if ((ascending && val1 < val2) || (!ascending && val1 > val2))
                    {
                        shared[tid] = val2;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Now scan remaining elements and maintain top-k
    for (int i = start + k; i < end; i++)
    {
        if (tid == 0)
        {
            float newVal = input[i];
            if (newVal > shared[k - 1])
            {
                // Insert and maintain sorted order
                shared[k - 1] = newVal;
                for (int j = k - 2; j >= 0; j--)
                {
                    if (shared[j] < shared[j + 1])
                    {
                        float temp = shared[j];
                        shared[j] = shared[j + 1];
                        shared[j + 1] = temp;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write local top-k to global memory
    if (tid < k && blockIdx.x * k + tid < gridDim.x * k)
    {
        output[blockIdx.x * k + tid] = shared[tid];
    }
}

// Merge sorted sequences
__global__ void mergeTopK(float *temp, float *output, int numBlocks, int k)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    // Load all block results into shared memory
    if (tid < numBlocks * k && tid < k * 32)
    { // Limit to reasonable size
        shared[tid] = temp[tid];
    }
    __syncthreads();

    // Simple parallel merge - find top k
    if (tid < k)
    {
        float maxVal = -FLT_MAX;
        int maxIdx = -1;

        for (int round = 0; round < k; round++)
        {
            if (tid == round)
            {
                maxVal = -FLT_MAX;
                maxIdx = -1;

                for (int i = 0; i < min(numBlocks * k, k * 32); i++)
                {
                    if (shared[i] > maxVal)
                    {
                        maxVal = shared[i];
                        maxIdx = i;
                    }
                }

                output[round] = maxVal;
                if (maxIdx >= 0)
                {
                    shared[maxIdx] = -FLT_MAX; // Mark as used
                }
            }
            __syncthreads();
        }
    }
}

// Kernel for top-k selection
__global__ void topKKernel(const float *input, int N, int k, float *output)
{
    // For small k, use a single block approach
    if (k <= 1024)
    {
        if (blockIdx.x == 0)
        {
            extern __shared__ float shared[];

            int tid = threadIdx.x;

            // Initialize with smallest values
            if (tid < k)
            {
                shared[tid] = -FLT_MAX;
            }
            __syncthreads();

            // Process all elements sequentially (single-threaded for correctness)
            if (tid == 0)
            {
                for (int idx = 0; idx < N; idx++)
                {
                    float val = input[idx];

                    // Find position to insert (maintain descending order)
                    for (int i = 0; i < k; i++)
                    {
                        if (val > shared[i])
                        {
                            // Shift elements right and insert
                            for (int j = k - 1; j > i; j--)
                            {
                                shared[j] = shared[j - 1];
                            }
                            shared[i] = val;
                            break;
                        }
                    }
                }
            }
            __syncthreads();

            // Write output
            if (tid < k)
            {
                output[tid] = shared[tid];
            }
        }
    }
    else
    {
        // For large k, use parallel bitonic sort approach
        extern __shared__ float shared[];
        int tid = threadIdx.x;
        int elementsPerThread = (N + blockDim.x - 1) / blockDim.x;

        // Each thread loads multiple elements
        for (int i = 0; i < elementsPerThread && tid * elementsPerThread + i < N; i++)
        {
            int idx = tid * elementsPerThread + i;
            if (idx < N)
            {
                shared[idx] = input[idx];
            }
        }
        __syncthreads();

        // Parallel bitonic sort for the entire array
        int sortSize = nextPowerOf2(N);
        for (int size = 2; size <= sortSize; size <<= 1)
        {
            for (int stride = size >> 1; stride > 0; stride >>= 1)
            {
                int idx = tid;
                while (idx < N)
                {
                    int partner = idx ^ stride;
                    if (partner < N && partner > idx)
                    {
                        bool ascending = ((idx & size) == 0);
                        float val1 = shared[idx];
                        float val2 = shared[partner];

                        if ((ascending && val1 < val2) || (!ascending && val1 > val2))
                        {
                            shared[idx] = val2;
                            shared[partner] = val1;
                        }
                    }
                    idx += blockDim.x;
                }
                __syncthreads();
            }
        }

        // Write top k elements to output
        if (tid < k)
        {
            output[tid] = shared[tid];
        }
    }
}

// Multi-block bitonic sort for very large arrays
__global__ void largeBitonicSort(float *data, int N, int j, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = i ^ j;

    if (ixj > i && i < N && ixj < N)
    {
        if ((i & k) == 0)
        {
            // Sort descending (we want largest first)
            if (data[i] < data[ixj])
            {
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
        else
        {
            // Sort ascending in this part
            if (data[i] > data[ixj])
            {
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// Host function that takes device pointers
extern "C" void solve(const float *input, float *output, int N, int k)
{
    // For very large arrays, use multi-block bitonic sort
    if (N > 10000)
    {
        // Allocate temporary array for sorting
        float *temp_array;
        cudaMalloc(&temp_array, N * sizeof(float));
        cudaMemcpy(temp_array, input, N * sizeof(float), cudaMemcpyDeviceToDevice);

        // Calculate grid dimensions
        int blockSize = BLOCK_SIZE;
        int gridSize = (N + blockSize - 1) / blockSize;

        // Bitonic sort - full sort of the array
        int sortSize = nextPowerOf2(N);
        for (int size = 2; size <= sortSize; size <<= 1)
        {
            for (int stride = size >> 1; stride > 0; stride >>= 1)
            {
                largeBitonicSort<<<gridSize, blockSize>>>(temp_array, N, stride, size);
                cudaDeviceSynchronize();
            }
        }

        // Copy top k elements to output
        cudaMemcpy(output, temp_array, k * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaFree(temp_array);
        return;
    }

    // For smaller arrays, use the existing kernel approach
    int blockSize = min(BLOCK_SIZE, max(k, 32));
    int sharedMemSize = (k <= 1024) ? k * sizeof(float) : min((int)(N * sizeof(float)), 48000);

    // Launch kernel
    topKKernel<<<1, blockSize, sharedMemSize>>>(input, N, k, output);

    // Wait for kernel to complete
    cudaDeviceSynchronize();
}

// Host wrapper function for testing
void topKSelection(float *h_input, int N, int k, float *h_output)
{
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, k * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Determine block and grid sizes
    int blockSize = min(BLOCK_SIZE, k);

    // Launch host function
    solve(d_input, d_output, N, k);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, k * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test function
int main()
{
    // Example 1
    float input1[] = {1.0, 5.0, 3.0, 2.0, 4.0};
    int N1 = 5, k1 = 3;
    float output1[3];

    topKSelection(input1, N1, k1, output1);

    printf("Example 1 output: ");
    for (int i = 0; i < k1; i++)
    {
        printf("%.1f ", output1[i]);
    }
    printf("\n");

    // Example 2
    float input2[] = {7.2, -1.0, 3.3, 8.8, 2.2};
    int N2 = 5, k2 = 2;
    float output2[2];

    topKSelection(input2, N2, k2, output2);

    printf("Example 2 output: ");
    for (int i = 0; i < k2; i++)
    {
        printf("%.1f ", output2[i]);
    }
    printf("\n");

    // Example 3: k = N case (this was slow before)
    float input3[] = {3.1, 1.5, 4.2, 2.8, 5.0, 0.9, 3.7, 2.1};
    int N3 = 8, k3 = 8;
    float output3[8];

    printf("Testing k = N case...\n");
    topKSelection(input3, N3, k3, output3);

    printf("Example 3 output (k=N): ");
    for (int i = 0; i < k3; i++)
    {
        printf("%.1f ", output3[i]);
    }
    printf("\n");

    // Example 4: Large scale test K = N = 100M
    printf("Testing large scale K=N=100M case...\n");
    int N4 = 100000000, k4 = 100000000;

    // Allocate large arrays
    float *input4 = (float *)malloc(N4 * sizeof(float));
    float *output4 = (float *)malloc(k4 * sizeof(float));

    if (input4 && output4)
    {
        // Initialize with random-like values for testing
        for (int i = 0; i < N4; i++)
        {
            input4[i] = (float)((i * 17 + 23) % 1000000) / 1000.0f;
        }

        printf("Starting large scale test...\n");
        topKSelection(input4, N4, k4, output4);

        printf("Large scale test completed. First 10 results: ");
        for (int i = 0; i < 10; i++)
        {
            printf("%.5f ", output4[i]);
        }
        printf("\nLast 10 results: ");
        for (int i = k4 - 10; i < k4; i++)
        {
            printf("%.5f ", output4[i]);
        }
        printf("\n");

        free(input4);
        free(output4);
    }
    else
    {
        printf("Failed to allocate memory for large test\n");
    }

    return 0;
}
