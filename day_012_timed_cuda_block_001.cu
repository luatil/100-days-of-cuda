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

        f32 Milliseconds = 0;
        cudaEventElapsedTime(&Milliseconds, Start, Stop);

        fprintf(stdout, "%s execution time: %f ms\n", Label, Milliseconds);
        cudaEventDestroy(Start);
        cudaEventDestroy(Stop);
    }
};

#define TimedCudaBlock(Name) cuda_profile_block Name

#define NameConcat2(A, B) A##B
#define NameConcat(A, B) NameConcat2(A, B)
#define TimeCudaBlock(Name) cuda_profile_block NameConcat(Block, __LINE__)(Name, __COUNTER__ + 1);
#define TimeCudaFunction TimeCudaBlock(__func__)
