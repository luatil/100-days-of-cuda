#include <float.h>
#include <stdio.h>

#define REPETITIONS 100

static void MeasureHtoD(float *H, float *D, size_t Bytes, const char *Label)
{
    cudaEvent_t Start, Stop;
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);

    double Max = 0.0, Min = DBL_MAX, Sum = 0.0;

    for (int R = 0; R < REPETITIONS; R++)
    {
        cudaEventRecord(Start);
        cudaMemcpy(D, H, Bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(Stop);
        cudaEventSynchronize(Stop);

        float Ms = 0;
        cudaEventElapsedTime(&Ms, Start, Stop);
        double GBs = (Bytes / 1e9) / (Ms / 1e3);

        if (GBs > Max)
            Max = GBs;
        if (GBs < Min)
            Min = GBs;
        Sum += GBs;
    }

    printf("HtoD %-20s  Max %6.2f  Min %6.2f  Avg %6.2f  GB/s\n", Label, Max, Min, Sum / REPETITIONS);

    cudaEventDestroy(Start);
    cudaEventDestroy(Stop);
}

static void MeasureDtoH(float *H, float *D, size_t Bytes, const char *Label)
{
    cudaEvent_t Start, Stop;
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);

    double Max = 0.0, Min = DBL_MAX, Sum = 0.0;

    for (int R = 0; R < REPETITIONS; R++)
    {
        cudaEventRecord(Start);
        cudaMemcpy(H, D, Bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(Stop);
        cudaEventSynchronize(Stop);

        float Ms = 0;
        cudaEventElapsedTime(&Ms, Start, Stop);
        double GBs = (Bytes / 1e9) / (Ms / 1e3);

        if (GBs > Max)
            Max = GBs;
        if (GBs < Min)
            Min = GBs;
        Sum += GBs;
    }

    printf("DtoH %-20s  Max %6.2f  Min %6.2f  Avg %6.2f  GB/s\n", Label, Max, Min, Sum / REPETITIONS);

    cudaEventDestroy(Start);
    cudaEventDestroy(Stop);
}

int main()
{
    const size_t Bytes = 512ULL * 1024 * 1024; // 512 MB
    const size_t N = Bytes / sizeof(float);

    float *D;
    cudaMalloc(&D, Bytes);

    printf("Transfer size: %zu MB  |  Repetitions: %d\n\n", Bytes / (1024 * 1024), REPETITIONS);
    printf("%-4s %-20s  %10s  %10s  %10s\n", "Dir", "Type", "Max GB/s", "Min GB/s", "Avg GB/s");
    printf("----------------------------------------------------------------------\n");

    // --- Pageable memory ---
    float *Pageable = (float *)malloc(Bytes);
    for (size_t I = 0; I < N; I++)
        Pageable[I] = 1.0f;

    MeasureHtoD(Pageable, D, Bytes, "(pageable)");
    MeasureDtoH(Pageable, D, Bytes, "(pageable)");

    free(Pageable);

    printf("----------------------------------------------------------------------\n");

    // --- Pinned memory ---
    float *Pinned;
    cudaMallocHost(&Pinned, Bytes);
    for (size_t I = 0; I < N; I++)
        Pinned[I] = 1.0f;

    MeasureHtoD(Pinned, D, Bytes, "(pinned)");
    MeasureDtoH(Pinned, D, Bytes, "(pinned)");

    cudaFreeHost(Pinned);
    cudaFree(D);
}
