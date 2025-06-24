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

__global__ void PrintHistogram(int* histogram, int num_bins)
{
    printf("HERE");
    for(int i = 0; i < num_bins; i++)
    {
        printf("%d ", histogram[i]);
    }
    printf("\n");
}

int main()
{
    puts("Here");
    const int input[] = {0, 1, 2, 1, 0};
    const int N = sizeof(input) / sizeof(input[0]);
    const int num_bins = 3;

    int *d_input;
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    int *d_histogram;
    cudaMalloc((void**)&d_histogram, num_bins * sizeof(int));
    cudaMemset(d_histogram, 0, num_bins * sizeof(int));

    solve(d_input, d_histogram, N, num_bins);
    cudaDeviceSynchronize();

    int *h_histogram = (int*)malloc(num_bins * sizeof(int));
    cudaMemcpy(h_histogram, d_histogram, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Histogram: ");
    for(int i = 0; i < num_bins; i++) {
        printf("%d ", h_histogram[i]);
    }
    printf("\n");

    free(h_histogram);
    cudaFree(d_histogram);
    cudaFree(d_input);
}
