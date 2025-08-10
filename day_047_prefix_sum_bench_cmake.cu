#include <nvbench/nvbench.cuh>
#include <stdio.h>

__global__ void PrefixSumNaive(float *X, int N)
{
    for (int I = 1; I < N; I++)
    {
        X[I] += X[I - 1];
    }
}

void KernelGenerator(nvbench::state &State)
{
    const int N = 1024 * 8;
    float *A = (float *)malloc(sizeof(float) * N);

    State.add_element_count(N, "NumElements");
    State.add_global_memory_reads<nvbench::int32_t>(N, "DataSize");
    State.add_global_memory_writes<nvbench::int32_t>(N);
    State.exec([&](nvbench::launch &Launch) { PrefixSumNaive<<<1, 1, 0, Launch.get_stream()>>>(A, N); });

    free(A);
}

NVBENCH_BENCH(KernelGenerator);

/*
Got the following results:
| NumElements |  DataSize  | Samples |  CPU Time  | Noise  |  GPU Time  | Noise  | Elem/s  | GlobalMem BW | BWUtil |
Samples | Batch GPU  |
|-------------|------------|---------|------------|--------|------------|--------|---------|--------------|--------|---------|------------|
|        8192 | 32.000 KiB |   2000x | 260.780 us | 29.85% | 250.626 us | 30.80% | 32.686M | 261.489 MB/s |  0.07% |
3630x | 137.824 us |
*/
