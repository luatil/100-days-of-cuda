#include <stdio.h>

typedef unsigned int u32;
typedef unsigned int b32;

#define MAX_VALUE 100000000

// Inplace ExclusiveScan
__global__ void ExclusiveScan(u32 *Array, u32 N)
{
    u32 Tid = blockDim.x * blockIdx.x + threadIdx.x;
}

__global__ void RadixSortIter(u32 *Input, u32 *Output, u32 *Bits, u32 N, u32 Iter)
{
    u32 Tid = blockDim.x * blockIdx.x + threadIdx.x;
    u32 Key = 0;
    u32 Bit = 0;

    if (Tid < N)
    {
        Key = Input[Tid];
        Bit = (Key >> Iter) & 1;
        Bits[Tid] = Bit;
    }

    // NOTE(luatil): This will need to be separated here into 3 different kernel calls.
    ExclusiveScan(Bits, N);

    if (Tid < N)
    {
        u32 NumberOfOnesBefore = Bits[Tid];
        u32 NumberOfOnesInTotal = Bits[N]; // Note here that Bits must have N+1 elements
        u32 Destination = (Bit == 0) ? (Tid - NumberOfOnesBefore) : (N - NumberOfOnesInTotal - NumberOfOnesBefore);

        Output[Destination] = Key;
    }
}

__host__ __device__ Swap(u32 **A, u32 **B)
{
    u32 *Temp = *A;
    *A = *B;
    *B = Temp;
}

// Input and Output are Device Pointers
void Sort(const u32 *Input, u32 *Output, u32 N)
{
    u32 *Temp;
    u32 *Bits;

    cudaMalloc(&Temp, sizeof(u32) * N);
    cudaMalloc(&Bits, sizeof(u32) * (N + 1)); // We use Bits[N] to get the total number of ones

    cudaMemcpy(&Temp, Input, sizeof(u32) * N, cudaMemcpyDeviceToDevice);

    Input = Temp;
    for (u32 Shift = 0; Shift < 32; Shift++)
    {
        RadixSortInit(Input, Bits, N, Shift);
        ExclusiveScan(Bits, N);
        RadixSortFinish(Input, Output, Bits, N);
        Swap(&Input, &Output);
    }

    cudaFree(&Temp);
    cudaFree(&Bits);
}

int main()
{
    u32 *Input = (u32 *)malloc(sizeof(u32) * MAX_VALUE); // Don't worry, virtual memory works
    u32 ReadElements = 0;

    for (u32 *It = Input; It - Input < MAX_VALUE; It++)
    {
        if (scanf("%u\n", It) != EOF)
        {
            ReadElements++;
        }
        else
        {
            break;
        }
    }

    u32 *Output = (u32 *)malloc(sizeof(u32) * ReadElements);

    {
        u32 *D_Input, *D_Output;
        cudaMalloc(&D_Input, sizeof(u32) * ReadElements);
        cudaMalloc(&D_Output, sizeof(u32) * ReadElements);

        cudaMemcpy(D_Input, Input, sizeof(u32) * ReadElements, cudaMemcpyHostToDevice);

        Sort(D_Input, D_Output, ReadElements);

        cudaMemcpy(Output, D_Output, sizeof(u32) * ReadElements, cudaMemcpyDeviceToHost);

        cudaFree(&D_Input);
        cudaFree(&D_Output);
    }

    for (u32 I = 0; I < ReadElements; I++)
    {
        printf("%d\n", Output[I]);
    }
    puts("");
}
