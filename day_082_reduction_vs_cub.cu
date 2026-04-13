/*
 * Day 082: Reduction Benchmark — Custom Kernels vs CUB
 *
 * Algorithms:
 * - CoarsenedMultiBlock: scalar loads, shared memory tree reduction (day 35 baseline)
 * - WarpShuffleReduce:   float4 loads + __shfl_down_sync intra-warp reduction
 * - CubReduce:           cub::DeviceReduce::Sum (SOTA reference)
 *
 * CUB on SM86 (RTX 3060) uses its Policy600 (Pascal tuning, no SM86 entry):
 *   256 threads, 16 items/thread, float4 loads, warp reductions, LOAD_LDG.
 * WarpShuffleReduce mirrors those choices:
 *   COARSE_FACTOR=4 float4 loads per thread → 16 scalars/thread.
 *
 * Assumes N is divisible by 4 (holds for all power-of-two sizes tested).
 */

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <stdint.h>

typedef float    f32;
typedef uint32_t u32;

// ---------------------------------------------------------------------------
// Kernel implementations
// ---------------------------------------------------------------------------

// CoarsenedMultiBlock (day 35 baseline):
//   - scalar 32-bit loads
//   - shared memory tree reduction (log2(BLOCK_DIM) __syncthreads barriers)
//   - one atomicAdd per block
template <int BLOCK_DIM, int COARSE_FACTOR>
__global__ void CoarsenedMultiBlock(const f32 *Input, f32 *Output, int N)
{
    __shared__ f32 Shared[BLOCK_DIM];

    const int TID = COARSE_FACTOR * blockDim.x * blockIdx.x + threadIdx.x;
    const int TX  = threadIdx.x;

    Shared[TX] = 0.0f;
    for (int I = 0; I < COARSE_FACTOR; I++)
    {
        if (TID + blockDim.x * I < N)
            Shared[TX] += Input[TID + blockDim.x * I];
    }
    __syncthreads();

    for (int Stride = blockDim.x / 2; Stride > 0; Stride /= 2)
    {
        if (TX < Stride)
            Shared[TX] += Shared[TX + Stride];
        __syncthreads();
    }

    if (TX == 0)
        atomicAdd(Output, Shared[0]);
}

// WarpShuffleReduce:
//   - float4 (128-bit) loads: 4x fewer memory transactions than scalar
//   - __shfl_down_sync: warp reduction with no shared memory and no barriers
//   - shared memory only for inter-warp communication (BLOCK_DIM/32 words)
//   - one atomicAdd per block (same as CoarsenedMultiBlock)
//
// With COARSE_FACTOR=4: each thread reads 4 float4 = 16 scalars, matching
// CUB's Policy600 tuning of 16 items/thread.
template <int BLOCK_DIM, int COARSE_FACTOR>
__global__ void WarpShuffleReduce(const f32 *Input, f32 *Output, int N)
{
    static_assert(BLOCK_DIM % 32 == 0, "BLOCK_DIM must be a multiple of warp size");

    const int VecN = N / 4;  // number of float4 elements
    const int TID  = (COARSE_FACTOR * blockDim.x) * blockIdx.x + threadIdx.x;
    const int TX   = threadIdx.x;

    const float4 *InputVec = reinterpret_cast<const float4 *>(Input);

    // Phase 1: thread-local accumulation using COARSE_FACTOR float4 loads.
    // Each load fetches 128 bits — 4x wider than a scalar load.
    f32 Sum = 0.0f;
    #pragma unroll
    for (int I = 0; I < COARSE_FACTOR; I++)
    {
        int VecIdx = TID + blockDim.x * I;
        if (VecIdx < VecN)
        {
            float4 V = InputVec[VecIdx];
            Sum += V.x + V.y + V.z + V.w;
        }
    }

    // Phase 2: warp reduction via shuffle — no shared memory, no barriers.
    // After this loop, lane 0 of every warp holds that warp's partial sum.
    #pragma unroll
    for (int Offset = 16; Offset > 0; Offset >>= 1)
        Sum += __shfl_down_sync(0xFFFFFFFF, Sum, Offset);

    // Phase 3: write one partial sum per warp into shared memory.
    // Only BLOCK_DIM/32 words used — e.g. 8 words for a 256-thread block.
    __shared__ f32 WarpSums[BLOCK_DIM / 32];
    if (TX % 32 == 0)
        WarpSums[TX / 32] = Sum;
    __syncthreads();

    // Phase 4: first warp reduces the BLOCK_DIM/32 warp partial sums.
    if (TX < 32)
    {
        Sum = (TX < BLOCK_DIM / 32) ? WarpSums[TX] : 0.0f;
        #pragma unroll
        for (int Offset = 16; Offset > 0; Offset >>= 1)
            Sum += __shfl_down_sync(0xFFFFFFFF, Sum, Offset);

        if (TX == 0)
            atomicAdd(Output, Sum);
    }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

void ReductionBenchmark(nvbench::state &State)
{
    const auto N          = static_cast<int>(State.get_int64("Elements"));
    const auto ALGORITHM  = State.get_string("Algorithm");
    const auto BLOCK_SIZE = static_cast<int>(State.get_int64("BlockSize"));

    const size_t InputBytes = N * sizeof(f32);

    f32 *DevInput  = nullptr;
    f32 *DevOutput = nullptr;
    cudaMalloc(&DevInput,  InputBytes);
    cudaMalloc(&DevOutput, sizeof(f32));

    std::vector<f32> HostInput(N, 1.0f);
    cudaMemcpy(DevInput, HostInput.data(), InputBytes, cudaMemcpyHostToDevice);

    // COARSE_FACTOR=4 for both custom kernels.
    // CoarsenedMultiBlock: each block covers BLOCK_SIZE * 4 scalars.
    // WarpShuffleReduce:   each block covers BLOCK_SIZE * 4 float4 = BLOCK_SIZE * 16 scalars.
    const int BlocksPerGridCoarsened = (N + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);
    const int VecN                   = N / 4;
    const int BlocksPerGridWS        = (VecN + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);

    void   *CubTemp      = nullptr;
    size_t  CubTempBytes = 0;
    cub::DeviceReduce::Sum(CubTemp, CubTempBytes, DevInput, DevOutput, N);
    cudaMalloc(&CubTemp, CubTempBytes);

    State.add_element_count(N, "Elements");
    State.add_global_memory_reads<f32>(N, "InputReads");
    State.add_global_memory_writes<f32>(1, "OutputWrites");

    State.exec([&](nvbench::launch &Launch) {
        cudaStream_t Stream = Launch.get_stream();

        if (ALGORITHM != "CubReduce")
            cudaMemsetAsync(DevOutput, 0, sizeof(f32), Stream);

        if (ALGORITHM == "CoarsenedMultiBlock")
        {
            if      (BLOCK_SIZE == 128)
                CoarsenedMultiBlock<128, 4><<<BlocksPerGridCoarsened, BLOCK_SIZE, 0, Stream>>>(DevInput, DevOutput, N);
            else if (BLOCK_SIZE == 256)
                CoarsenedMultiBlock<256, 4><<<BlocksPerGridCoarsened, BLOCK_SIZE, 0, Stream>>>(DevInput, DevOutput, N);
            else if (BLOCK_SIZE == 512)
                CoarsenedMultiBlock<512, 4><<<BlocksPerGridCoarsened, BLOCK_SIZE, 0, Stream>>>(DevInput, DevOutput, N);
            else
                CoarsenedMultiBlock<1024, 4><<<BlocksPerGridCoarsened, BLOCK_SIZE, 0, Stream>>>(DevInput, DevOutput, N);
        }
        else if (ALGORITHM == "WarpShuffleReduce")
        {
            if      (BLOCK_SIZE == 128)
                WarpShuffleReduce<128, 4><<<BlocksPerGridWS, BLOCK_SIZE, 0, Stream>>>(DevInput, DevOutput, N);
            else if (BLOCK_SIZE == 256)
                WarpShuffleReduce<256, 4><<<BlocksPerGridWS, BLOCK_SIZE, 0, Stream>>>(DevInput, DevOutput, N);
            else if (BLOCK_SIZE == 512)
                WarpShuffleReduce<512, 4><<<BlocksPerGridWS, BLOCK_SIZE, 0, Stream>>>(DevInput, DevOutput, N);
            else
                WarpShuffleReduce<1024, 4><<<BlocksPerGridWS, BLOCK_SIZE, 0, Stream>>>(DevInput, DevOutput, N);
        }
        else if (ALGORITHM == "CubReduce")
        {
            cub::DeviceReduce::Sum(CubTemp, CubTempBytes, DevInput, DevOutput, N, Stream);
        }
    });

    cudaFree(DevInput);
    cudaFree(DevOutput);
    cudaFree(CubTemp);
}

NVBENCH_BENCH(ReductionBenchmark)
    .add_int64_power_of_two_axis("Elements",  nvbench::range(20, 25, 2))
    .add_string_axis("Algorithm", {"CoarsenedMultiBlock", "WarpShuffleReduce", "CubReduce"})
    .add_int64_power_of_two_axis("BlockSize", nvbench::range(7, 10, 1))
    .set_timeout(30);
