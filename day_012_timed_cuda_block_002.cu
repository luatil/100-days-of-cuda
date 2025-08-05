#include "day_001_macros.h"
#include <cuda_runtime.h>
#include <stdint.h>

typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;
typedef double f64;

struct cuda_profile_block
{
    char const *Label;
    u32 GlobalIndex;
    u64 BytesRead;
    u64 BytesWritten;
    f32 FlopsPerByte;
    cudaEvent_t Start;
    cudaEvent_t Stop;

    cuda_profile_block(const char *Label, u32 GlobalIndex, u64 BytesRead, u64 BytesWritten, f32 FlopsPerByte)
    {
        cudaEventCreate(&Start);
        cudaEventCreate(&Stop);
        cudaEventRecord(Start);

        Label = Label;
        GlobalIndex = GlobalIndex;
        BytesRead = BytesRead;
        BytesWritten = BytesWritten;
        FlopsPerByte = FlopsPerByte;
    }

    ~cuda_profile_block(void)
    {
        cudaEventRecord(Stop);
        cudaEventSynchronize(Stop);

        f32 Milliseconds = 0;
        cudaEventElapsedTime(&Milliseconds, Start, Stop);

        f32 Megabyte = 1024.0f * 1024.0f;
        f32 Gigabyte = Megabyte * 1024.0f;

        u64 BytesProcessed = BytesRead + BytesWritten;
        f32 TotalFlops = FlopsPerByte * BytesProcessed;
        f32 FlopsPerSecond = (TotalFlops / Milliseconds) * 1000.0f;

        f32 Bandwidth = (BytesProcessed / Milliseconds) * 1000.0f;

        fprintf(stdout, "%s | Execution Time: %f ms\n", Label, Milliseconds);

        if (BytesProcessed && FlopsPerByte)
        {
            fprintf(stdout, "%s | Bytes Processed: %.2f Mb\n", Label, BytesProcessed / Megabyte);
            fprintf(stdout, "%s | Effective Bandwidth: %.4f Gb/s\n", Label, Bandwidth / Gigabyte);
            fprintf(stdout, "%s | Compute Throughput: %.4f GFLOPS/s\n", Label, FlopsPerSecond / Gigabyte);
        }

        fprintf(stdout, "-------------------------\n");

        cudaEventDestroy(Start);
        cudaEventDestroy(Stop);
    }
};

#define NAME_CONCAT2(A, B) A##B
#define NAME_CONCAT(A, B) NameConcat2(A, B)
#define TIME_CUDA_BLOCK(Name) cuda_profile_block NameConcat(Block, __LINE__)(Name, __COUNTER__ + 1, 0, 0, 0.0f);
#define TIME_CUDA_BANDWIDTH(Name, BytesRead_, BytesWritten_, FlopsPerByte_)                                              \
    cuda_profile_block NameConcat(Block, __LINE__)(Name, __COUNTER__ + 1, BytesRead_, BytesWritten_, FlopsPerByte_);
#define TIME_CUDA_FUNCTION TimeCudaBlock(__func__)
