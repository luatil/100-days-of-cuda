#include <stdio.h>

// #include "day_023_histogram_00.cu"
#include "day_023_histogram_01.cu"

/*
 * Write a GPU program that computes the histogram of an array of 32-bit
 * integers. The histogram should count the number of occurrences of each
 * integer value in the range [0, num_bins). You are given an input array
 * input of length N and the number of bins num_bins.

 * The result should be an array of integers of length num_bins, where
 * each element represents the count of occurrences of its corresponding
 * index in the input array.
 */

/*
 * Input: input = [0, 1, 2, 1, 0],  N = 5, num_bins = 3
 * Output: [2, 2, 1]
 */

/*
 * Input: input = [3, 3, 3, 3], N = 4, num_bins = 5
 * Output: [0, 0, 0, 4, 0]
 */

/*
 * 1 ≤ N ≤ 100,000,000
 * 0 ≤ input[i] < num_bins
 * 1 ≤ num_bins ≤ 1024
 */

/*
 * // input, histogram are device pointers
 * void solve(const int* input, int* histogram, int N, int num_bins) {
 *
 * }
 */

__global__ void PrintHistogram(int *Histogram, int NumBins)
{
    printf("HERE");
    for (int I = 0; I < NumBins; I++)
    {
        printf("%d ", Histogram[I]);
    }
    printf("\n");
}

int main()
{
    puts("Here");
    const int INPUT[] = {0, 1, 2, 1, 0};
    const int N = sizeof(INPUT) / sizeof(INPUT[0]);
    const int NUM_BINS = 3;

    int *DInput;
    cudaMalloc((void **)&DInput, N * sizeof(int));
    cudaMemcpy(DInput, INPUT, N * sizeof(int), cudaMemcpyHostToDevice);

    int *DHistogram;
    cudaMalloc((void **)&DHistogram, NUM_BINS * sizeof(int));
    cudaMemset(DHistogram, 0, NUM_BINS * sizeof(int));

    solve(DInput, DHistogram, N, NUM_BINS);
    cudaDeviceSynchronize();

    int *HHistogram = (int *)malloc(NUM_BINS * sizeof(int));
    cudaMemcpy(HHistogram, DHistogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Histogram: ");
    for (int I = 0; I < NUM_BINS; I++)
    {
        printf("%d ", HHistogram[I]);
    }
    printf("\n");

    free(HHistogram);
    cudaFree(DHistogram);
    cudaFree(DInput);
}
