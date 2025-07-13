/*
 * NAME
 * 	gpusort - sorts numbers with the GPU
 *
 * SYNOPSIS
 * 	gpusort
 *
 * DESCRIPTION
 * 	Sorts unsigned integer from stdin to stdout using the GPU.
 *
 * USAGE
 * 	seq 1 10000 | shuf | gpusort
 *
 */
#include <stdio.h>

#define MAX_LINES (1 << 24)

#define SORT_FUNC 2

static int Lines[MAX_LINES]; // 17M

__device__ void Swap(int *X, int I, int J)
{
    int Temp = X[I];
    X[I] = X[J];
    X[J] = Temp;
}

#if SORT_FUNC == 0
__global__ void SortKernel(int *X, int N)
{
    for (int I = 0; I < N; I++)
    {
        for (int J = 0; J < N; J++)
        {
            if (X[I] < X[J])
            {
                Swap(X, I, J);
            }
        }
    }
}
#elif SORT_FUNC == 1
// TODO(luatil): Simple Radix Sort only works for unsigned values
// This is a naive version that only works with <<<1,1>>>
__global__ void SortKernel(int *X, int *Temp, int N, int Bit)
{
    // First is counting sort on X for Bit
    int Bucket[2] = {0};

    for (int I = 0; I < N; I++)
    {
        Bucket[(X[I] >> Bit) & 1]++;
    }

    // Now exclusive sum on Bucket
    // Which for the simple binary case is:
    Bucket[1] = Bucket[0];
    Bucket[0] = 0;

    for (int I = 0; I < N; I++)
    {
        int CurrentBit = (X[I] >> Bit) & 1;
        Temp[Bucket[CurrentBit]++] = X[I];
    }
}
#elif SORT_FUNC == 2
#endif

// X is a device ptr
static void Sort(int *X, int N)
{
#if SORT_FUNC == 0
    SortKernel<<<1, 1>>>(X, N);
#elif SORT_FUNC == 1
    int *Temp;
    cudaMalloc(&Temp, sizeof(int) * N);

    for (int Bit = 0; Bit < 31; Bit++) // Change to 32 for unsigned int
    {
        SortKernel<<<1, 1>>>(X, Temp, N, Bit);
        cudaMemcpy(X, Temp, sizeof(int) * N, cudaMemcpyDeviceToDevice);
    }

    cudaFree(&Temp);
#elif SORT_FUNC == 2
#endif
}

int main()
{
    int Num = 0;
    int LineIter = 0;

    while (scanf("%d\n", &Num) != EOF)
    {
        Lines[LineIter++] = Num;

        if (LineIter >= MAX_LINES)
        {
            fprintf(stderr, "Maximun number of lines reached: [%d]\n", MAX_LINES);
            exit(1);
        }
    }

    int *GLines;
    cudaMalloc(&GLines, sizeof(int) * LineIter);
    cudaMemcpy(GLines, Lines, sizeof(int) * LineIter, cudaMemcpyHostToDevice);

    Sort(GLines, LineIter);

    cudaMemcpy(Lines, GLines, sizeof(int) * LineIter, cudaMemcpyDeviceToHost);

    for (int I = 0; I < LineIter; I++)
    {
        printf("%d\n", Lines[I]);
    }
}

// How Parallel Radix Sort Works:
//
// Parallel Radix Sort is an application of a stable sorting algorithm
// over consecutive bit indexes of an unsigned int array. Let's us define
// a notion of RadixSortIter_i which is the application of RadixSortIter
// over the bit i.
//
// Side Note: While it is true that RadixSort can be applied only over
// a single bit, this is an inneficient way of doing it. The algorithm
// presented here can be generalized to operate over a large number of
// buckets by using bit partitions.
//
// Consider the following sequence of unsigned 4 bit integers:
//
// 0001 | 0011 | 0110 | 0101 | 0110 | 0001 | 1010 | 0111
//
// We will first consider an application of the algorithm for bit 0.
//
// Histogramming for bit 0 (LSB):
//
// Input: 0001 | 0011 | 0110 | 0101 | 0110 | 0001 | 1010 | 0111
// Bit 0:   1  |  1   |  0   |  1   |  0   |  1   |  0   |  1
//
// Count[0] = 3
// Count[1] = 5
//
// We then apply an exclusive scan procedure over the counts:
//
// Count[0] = 0
// Count[1] = 3
//
// And them we apply an expand operation to populate a temporary
// array while incrementing the count of each bucket.
//
// Inputs:
// I = 0
// Input[I] = 0001
// Count[1] = 3
//
// Outputs:
// Count[1] = 4
// Temp[3] = 0001
//
// Inputs:
// I = 1
// Input[I] = 0011
// Count[1] = 4
//
// Outputs:
// Count[1] = 5
// Temp[4] = 0011
// ...
//
// Since each iteration applies a stable sort, if we apply this procedure for each Bit [0, 31]
// we will sort the entire array.
//
// Challenges:
//
// Histogramming and exclusive scan are well known parallel primitives. Therefore the main challenges
// of implementing this algorithm in a GPU friendly way are composing these primitives in an efficient
// way, and implementing the scatter operation. This operation looks sequential in nature, so at
// least in the beginning it does not look trivial to parallelize it.
//
// To approach this problem we can leverage the exclusive scan primitive to somehow "precompute"
// the target position for each element.
//
// Let's look at our input and how it processed each element of the input for the Bit 0 iteration.
//
// Input:  0001 | 0011 | 0110 | 0101 | 0110 | 0001 | 1010 | 0111
// Bit 0:    1  |  1   |  0   |  1   |  0   |  1   |  0   |  1
// Target:   3  |  4   |  0   |  5   |  1   |  6   |  2   |  7
// Output: 0110 | 0110 | 1010 | 0001 | 0011 | 0101 | 0001 | 0111
//
// We can see that the target values are composed of two intertwined monotonically incrementing
// sequences:
//
// 0:   .  |  .   |  0   |  .   |  1   |  .   |  2   |  .
// 1:   3  |  4   |  .   |  5   |  .   |  6   |  .   |  7
//
// And that they align with the last bit of the input.
//
// We can also notice that the number that starts the one sequence [3] is the number just
// after the last one in the zero sequence [2].
//
