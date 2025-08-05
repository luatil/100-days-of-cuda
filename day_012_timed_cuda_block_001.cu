#include <cuda_runtime.h>

typedef unsigned int u32;

struct cuda_profile_block
{
    char const *Label;
    u32 GlobalIndex;
    cudaEvent_t Start;
    cudaEvent_t Stop;

    cuda_profile_block(const char *Label, u32 GlobalIndex)
    {
        cudaEventCreate(&Start);
        cudaEventCreate(&Stop);
        cudaEventRecord(Start);

        Label = Label;
        GlobalIndex = GlobalIndex;
    }

    ~cuda_profile_block(void)
    {
        cudaEventRecord(Stop);
        cudaEventSynchronize(Stop);

        u32 Milliseconds = 0;
        cudaEventElapsedTime(&Milliseconds, Start, Stop);

        fprintf(stdout, "%s execution time: %f ms\n", Label, Milliseconds);
        cudaEventDestroy(Start);
        cudaEventDestroy(Stop);
    }
};

#define TIMED_CUDA_BLOCK(Name) cuda_profile_block Name

#define NAME_CONCAT2(A, B) A##B
#define NAME_CONCAT(A, B) NameConcat2(A, B)
#define TIME_CUDA_BLOCK(Name) cuda_profile_block NameConcat(Block, __LINE__)(Name, __COUNTER__ + 1);
#define TIME_CUDA_FUNCTION TimeCudaBlock(__func__)
