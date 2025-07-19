/*
 * Day 048: Comprehensive Reduction Benchmarking Suite
 *
 * Benchmarks multiple reduction algorithm implementations:
 * - TreeReduction (day 10): Optimized tree reduction without divergence
 * - NaiveSingleBlock (day 15-001): Simple stride-doubling reduction
 * - ImprovedSingleBlock (day 15-002): Stride-halving reduction
 * - SharedMemorySingleBlock (day 15-003): Single block with shared memory
 * - MultiBlockAtomic (day 15 multi-block): Multi-block with atomic aggregation
 * - CoarsenedMultiBlock (day 35): Multi-block with 4x coarsening factor
 */

#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <stdint.h>
#include <stdio.h>

// Type definitions following project conventions
typedef float f32;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t s32;

//
// REDUCTION KERNEL IMPLEMENTATIONS
//

// 1. TreeReduction (Day 10) - Optimized tree reduction without divergence
template <int BLOCK_SIZE> __global__ void TreeReduction(const f32 *Input, f32 *Output, int N)
{
    __shared__ f32 SharedData[BLOCK_SIZE];

    int Tid = blockIdx.x * blockDim.x + threadIdx.x;
    int Tx = threadIdx.x;

    // Load data into shared memory
    SharedData[Tx] = (Tid < N) ? Input[Tid] : 0.0f;
    __syncthreads();

    // Tree reduction - improved version without divergence
    for (int S = blockDim.x / 2; S > 0; S >>= 1)
    {
        if (Tx < S)
        {
            SharedData[Tx] += SharedData[Tx + S];
        }
        __syncthreads();
    }

    // Thread 0 writes result for this block
    if (Tx == 0)
    {
        Output[blockIdx.x] = SharedData[0];
    }
}

// 2. NaiveSingleBlock (Day 15-001) - Simple stride-doubling reduction
__global__ void NaiveSingleBlock(f32 *Input, f32 *Output, u32 N)
{
    u32 Tx = threadIdx.x;
    u32 Tid = Tx * 2;

    for (u32 Stride = 1; Stride <= blockDim.x; Stride *= 2)
    {
        if (Tx % Stride == 0)
        {
            Input[Tid] += Input[Tid + Stride];
        }
        __syncthreads();
    }

    if (Tx == 0)
    {
        *Output = Input[0];
    }
}

// 3. ImprovedSingleBlock (Day 15-002) - Stride-halving reduction
__global__ void ImprovedSingleBlock(f32 *Input, f32 *Output, u32 N)
{
    u32 Tid = threadIdx.x;

    for (u32 Stride = blockDim.x; Stride >= 1; Stride /= 2)
    {
        if (Tid < Stride)
        {
            Input[Tid] += Input[Tid + Stride];
        }
        __syncthreads();
    }

    if (Tid == 0)
    {
        *Output = Input[0];
    }
}

// 4. SharedMemorySingleBlock (Day 15-003) - Single block with shared memory
template <int BLOCK_DIM> __global__ void SharedMemorySingleBlock(f32 *Input, f32 *Output, u32 N)
{
    __shared__ f32 Shared[BLOCK_DIM];

    u32 Tid = threadIdx.x;

    Shared[Tid] = Input[Tid] + Input[Tid + BLOCK_DIM];

    for (u32 Stride = blockDim.x / 2; Stride >= 1; Stride /= 2)
    {
        __syncthreads();
        if (Tid < Stride)
        {
            Shared[Tid] += Shared[Tid + Stride];
        }
    }

    if (Tid == 0)
    {
        *Output = Shared[0];
    }
}

// 5. MultiBlockAtomic (Day 15 multi-block) - Multi-block with atomic aggregation
template <int BLOCK_DIM> __global__ void MultiBlockAtomic(f32 *Input, f32 *Output, u32 N)
{
    __shared__ f32 Shared[BLOCK_DIM];

    u32 Segment = 2 * blockDim.x * blockIdx.x;
    u32 Tid = Segment + threadIdx.x;
    u32 Tx = threadIdx.x;

    Shared[Tx] = 0.0f;
    if (Tid < N)
        Shared[Tx] += Input[Tid];
    if (Tid + BLOCK_DIM < N)
        Shared[Tx] += Input[Tid + BLOCK_DIM];

    for (u32 Stride = blockDim.x / 2; Stride >= 1; Stride /= 2)
    {
        __syncthreads();
        if (Tx < Stride)
        {
            Shared[Tx] += Shared[Tx + Stride];
        }
    }

    if (Tx == 0)
    {
        atomicAdd(Output, Shared[0]);
    }
}

// 6. CoarsenedMultiBlock (Day 35) - Multi-block with 4x coarsening factor
template <int BLOCK_DIM, int COARSE_FACTOR> __global__ void CoarsenedMultiBlock(const f32 *Input, f32 *Output, int N)
{
    __shared__ f32 Shared[BLOCK_DIM];

    const int Tid = COARSE_FACTOR * blockDim.x * blockIdx.x + threadIdx.x;
    const int Tx = threadIdx.x;

    Shared[Tx] = 0.0f;
    for (int I = 0; I < COARSE_FACTOR; I++)
    {
        if (Tid + blockDim.x * I < N)
        {
            Shared[Tx] += Input[Tid + blockDim.x * I];
        }
    }
    __syncthreads();

    for (int Stride = blockDim.x / 2; Stride > 0; Stride /= 2)
    {
        if (Tx < Stride)
        {
            Shared[Tx] += Shared[Tx + Stride];
        }
        __syncthreads();
    }

    if (Tx == 0)
    {
        atomicAdd(Output, Shared[0]);
    }
}

//
// NVBENCH BENCHMARK IMPLEMENTATION
//

void ReductionBenchmark(nvbench::state &State)
{
    const auto N = State.get_int64("Elements");
    const auto Algorithm = State.get_string("Algorithm");
    const auto BlockSize = State.get_int64("BlockSize");

    // Allocate memory
    const size_t Bytes = N * sizeof(f32);
    f32 *DeviceInput, *DeviceOutput, *DeviceTemp;
    cudaMalloc(&DeviceInput, Bytes);
    cudaMalloc(&DeviceOutput, sizeof(f32));
    cudaMalloc(&DeviceTemp, Bytes); // For algorithms that modify input

    // Initialize data with 1.0f for easy verification
    std::vector<f32> HostInput(N, 1.0f);
    cudaMemcpy(DeviceInput, HostInput.data(), Bytes, cudaMemcpyHostToDevice);

    // Configure metrics for bandwidth calculation
    State.add_element_count(N, "Elements");
    State.add_global_memory_reads<f32>(N, "InputReads");
    State.add_global_memory_writes<f32>(1, "OutputWrites");

    // Calculate grid dimensions
    const int BlocksPerGrid = (N + BlockSize - 1) / BlockSize;
    const int BlocksPerGridCoarsened = (N + BlockSize * 4 - 1) / (BlockSize * 4);

    State.exec([&](nvbench::launch &Launch) {
        // Reset output for atomic variants
        cudaMemset(DeviceOutput, 0, sizeof(f32));

        // Copy input to temp for algorithms that modify input
        cudaMemcpy(DeviceTemp, DeviceInput, Bytes, cudaMemcpyDeviceToDevice);

        if (Algorithm == "TreeReduction")
        {
            if (BlockSize == 128)
            {
                TreeReduction<128><<<BlocksPerGrid, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }
            else if (BlockSize == 256)
            {
                TreeReduction<256><<<BlocksPerGrid, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }
            else if (BlockSize == 512)
            {
                TreeReduction<512><<<BlocksPerGrid, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }
            else if (BlockSize == 1024)
            {
                TreeReduction<1024><<<BlocksPerGrid, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }

            // TreeReduction outputs per-block results, need final reduction
            if (BlocksPerGrid > 1)
            {
                int FinalThreads = (int)min(BlocksPerGrid, (int)BlockSize);
                if (BlockSize == 128)
                {
                    TreeReduction<128>
                        <<<1, FinalThreads, 0, Launch.get_stream()>>>(DeviceOutput, DeviceOutput, BlocksPerGrid);
                }
                else if (BlockSize == 256)
                {
                    TreeReduction<256>
                        <<<1, FinalThreads, 0, Launch.get_stream()>>>(DeviceOutput, DeviceOutput, BlocksPerGrid);
                }
                else if (BlockSize == 512)
                {
                    TreeReduction<512>
                        <<<1, FinalThreads, 0, Launch.get_stream()>>>(DeviceOutput, DeviceOutput, BlocksPerGrid);
                }
                else if (BlockSize == 1024)
                {
                    TreeReduction<1024>
                        <<<1, FinalThreads, 0, Launch.get_stream()>>>(DeviceOutput, DeviceOutput, BlocksPerGrid);
                }
            }
        }
        else if (Algorithm == "NaiveSingleBlock")
        {
            // Single block only - limit to reasonable size
            if (N <= 2048)
            {
                int ThreadsToUse = (int)min((int)(N / 2), (int)BlockSize);
                NaiveSingleBlock<<<1, ThreadsToUse, 0, Launch.get_stream()>>>(DeviceTemp, DeviceOutput, N);
            }
        }
        else if (Algorithm == "ImprovedSingleBlock")
        {
            // Single block only - limit to reasonable size
            if (N <= 2048)
            {
                int ThreadsToUse = (int)min((int)(N / 2), (int)BlockSize);
                ImprovedSingleBlock<<<1, ThreadsToUse, 0, Launch.get_stream()>>>(DeviceTemp, DeviceOutput, N);
            }
        }
        else if (Algorithm == "SharedMemorySingleBlock")
        {
            // Single block only - limit to reasonable size
            if (N <= 2048)
            {
                int ThreadsToUse = (int)min((int)(N / 2), (int)BlockSize);
                if (BlockSize == 128)
                {
                    SharedMemorySingleBlock<128>
                        <<<1, ThreadsToUse, 0, Launch.get_stream()>>>(DeviceTemp, DeviceOutput, N);
                }
                else if (BlockSize == 256)
                {
                    SharedMemorySingleBlock<256>
                        <<<1, ThreadsToUse, 0, Launch.get_stream()>>>(DeviceTemp, DeviceOutput, N);
                }
                else if (BlockSize == 512)
                {
                    SharedMemorySingleBlock<512>
                        <<<1, ThreadsToUse, 0, Launch.get_stream()>>>(DeviceTemp, DeviceOutput, N);
                }
                else if (BlockSize == 1024)
                {
                    SharedMemorySingleBlock<1024>
                        <<<1, ThreadsToUse, 0, Launch.get_stream()>>>(DeviceTemp, DeviceOutput, N);
                }
            }
        }
        else if (Algorithm == "MultiBlockAtomic")
        {
            if (BlockSize == 128)
            {
                MultiBlockAtomic<128>
                    <<<BlocksPerGrid, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }
            else if (BlockSize == 256)
            {
                MultiBlockAtomic<256>
                    <<<BlocksPerGrid, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }
            else if (BlockSize == 512)
            {
                MultiBlockAtomic<512>
                    <<<BlocksPerGrid, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }
            else if (BlockSize == 1024)
            {
                MultiBlockAtomic<1024>
                    <<<BlocksPerGrid, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }
        }
        else if (Algorithm == "CoarsenedMultiBlock")
        {
            if (BlockSize == 128)
            {
                CoarsenedMultiBlock<128, 4>
                    <<<BlocksPerGridCoarsened, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }
            else if (BlockSize == 256)
            {
                CoarsenedMultiBlock<256, 4>
                    <<<BlocksPerGridCoarsened, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }
            else if (BlockSize == 512)
            {
                CoarsenedMultiBlock<512, 4>
                    <<<BlocksPerGridCoarsened, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }
            else if (BlockSize == 1024)
            {
                CoarsenedMultiBlock<1024, 4>
                    <<<BlocksPerGridCoarsened, BlockSize, 0, Launch.get_stream()>>>(DeviceInput, DeviceOutput, N);
            }
        }
    });

    // Cleanup
    cudaFree(DeviceInput);
    cudaFree(DeviceOutput);
    cudaFree(DeviceTemp);
}

// NVBench configuration
NVBENCH_BENCH(ReductionBenchmark)
    .add_int64_power_of_two_axis("Elements", nvbench::range(10, 24, 2)) // 1K to 16M elements
    .add_string_axis("Algorithm", {"TreeReduction", "NaiveSingleBlock", "ImprovedSingleBlock",
                                   "SharedMemorySingleBlock", "MultiBlockAtomic", "CoarsenedMultiBlock"})
    .add_int64_power_of_two_axis("BlockSize", nvbench::range(7, 10, 1)) // 128 to 1024 threads
    .set_timeout(30);
