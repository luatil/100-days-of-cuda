#include <stdint.h>
#include <stdio.h>

typedef int32_t b32;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;

#include "day_001_macros.h"

#define Min(_a, _b) (_a < _b) ? _a : _b
#define Max(_a, _b) (_a > _b) ? _a : _b

enum test_mode
{
    TEST_MODE_UNINITIALIZED,
    TEST_MODE_TESTING,
    TEST_MODE_COMPLETED,
    TEST_MODE_ERROR
};

struct cuda_repetition_tester
{
    u64 TotalBytesRead;
    u64 TotalBytesWrote;
    f32 FlopsPerByte;

    f32 TotalTimeMs;
    u64 TotalCount; // Times Begin/End block called

    u64 TryForTimeMs; // Stop testing after reaching this time

    f32 MinTimeMs;
    f32 MaxTimeMs;
    f32 AvgTimeMs;

    test_mode Mode;

    cudaEvent_t CudaStart;
    cudaEvent_t CudaStop;
};

static void BeginTime(cuda_repetition_tester *Tester)
{
    cudaEventRecord(Tester->CudaStart);
}

static void EndTime(cuda_repetition_tester *Tester)
{
    cudaEventRecord(Tester->CudaStop);
    cudaEventSynchronize(Tester->CudaStop);

    f32 ElapsedTimeMS = 0.0f;

    if (cudaEventElapsedTime(&ElapsedTimeMS, Tester->CudaStart, Tester->CudaStop))
    {
        fprintf(stderr, "ERROR | cudaEventElapsedTime\n");
    }

    Tester->MinTimeMs = Min(Tester->MinTimeMs, ElapsedTimeMS);
    Tester->MaxTimeMs = Max(Tester->MaxTimeMs, ElapsedTimeMS);
    Tester->TotalTimeMs += ElapsedTimeMS;
}

static void StartTesting(cuda_repetition_tester *Tester, u64 TotalBytesRead = 0, u64 TotalBytesWrote = 0,
                         f32 FlopsPerByte = 0.0f, u32 SecondsToTry = 10)
{
    Tester->Mode = TEST_MODE_TESTING;
    Tester->TryForTimeMs = SecondsToTry * 1000.0f;
    Tester->MinTimeMs = 10000000.0f; // TODO(luatil): Set this to +inf
    Tester->TotalBytesRead = TotalBytesRead;
    Tester->TotalBytesWrote = TotalBytesWrote;
    Tester->FlopsPerByte = FlopsPerByte;
    Tester->TotalTimeMs = 0;

    cudaEventCreate(&Tester->CudaStart);
    cudaEventCreate(&Tester->CudaStop);
}

static void PrintResults(cuda_repetition_tester *Tester, const char *Label)
{
    f32 Megabyte = 1024.0f * 1024.0f;
    f32 Gigabyte = Megabyte * 1024.0f;

    u64 BytesProcessed = Tester->TotalBytesRead + Tester->TotalBytesWrote;
    f32 TotalFlops = Tester->FlopsPerByte * BytesProcessed;

    // Execution Time
    f32 MinEx = Tester->MinTimeMs;
    f32 AvgEx = Tester->TotalTimeMs / Tester->TotalCount;
    f32 MaxEx = Tester->MaxTimeMs;

    // Effective Bandwidth
    f32 MinEB = 1000 * (BytesProcessed / MaxEx) / Gigabyte;
    f32 AvgEB = 1000 * (BytesProcessed / AvgEx) / Gigabyte;
    f32 MaxEB = 1000 * (BytesProcessed / MinEx) / Gigabyte;

    // Compute Througput
    f32 MinCT = 1000 * (TotalFlops / MaxEx) / Gigabyte;
    f32 AvgCT = 1000 * (TotalFlops / AvgEx) / Gigabyte;
    f32 MaxCT = 1000 * (TotalFlops / MinEx) / Gigabyte;

    /*
     * VectorAddNaive:
     *
     *        BytesProcessed | 200mb
     *        Execution Time | Min: 1.00 ms Avg: 20.9 ms Max: 30.3993 ms
     *   Effective Bandwidth | Min: 1.000 Gb/s Avg: 2.000 Gb/s Max: 3.000 Gb/s
     *    Compute Throughput | Min: 1.000 GFLOPS/s Avg: 2.000 Gb/s Max: 3.000 Gb/s
     *
     * -----------------------
     */

    fprintf(stdout, "%s:\n", Label);
    fprintf(stdout, "\n");
    fprintf(stdout, "      Bytes Processed | %f mb\n", BytesProcessed / Megabyte);
    fprintf(stdout, "       Execution Time | Min: %.3f ms Avg: %.3f ms Max: %.3f ms\n", MinEx, AvgEx, MaxEx);
    fprintf(stdout, "  Effective Bandwidth | Min: %.3f Gb/s Avg: %.3f Gb/s Max: %.3f Gb/s\n", MinEB, AvgEB, MaxEB);
    fprintf(stdout, "   Compute Throughput | Min: %.3f GLOPS/s Avg: %.3f GFLOPS/s Max: %.3f GLOPS/s\n", MinCT, AvgCT,
            MaxCT);
    fprintf(stdout, "\n");
    fprintf(stdout, "-------------------------\n");
}

static b32 IsTesting(cuda_repetition_tester *Tester)
{
    if (Tester->Mode == TEST_MODE_TESTING)
    {
        // Calculate total time testing
        if (Tester->TotalTimeMs < Tester->TryForTimeMs)
        {
            Tester->TotalCount++;

            // Calculate progress percentage
            f32 Progress = (Tester->TotalTimeMs / Tester->TryForTimeMs) * 100.0f;
            u32 BarWidth = 50; // Width of the progress bar
            u32 FilledBars = (u32)((Progress / 100.0f) * BarWidth);

            // Print progress bar
            fprintf(stdout, "\rProgress: [");
            for (u32 i = 0; i < BarWidth; i++)
            {
                if (i < FilledBars)
                {
                    fprintf(stdout, "=");
                }
                else if (i == FilledBars)
                {
                    fprintf(stdout, ">");
                }
                else
                {
                    fprintf(stdout, " ");
                }
            }
            fprintf(stdout, "] %.1f%% | Min Time: %.3f ms | Count: %lu", Progress, Tester->MinTimeMs,
                    Tester->TotalCount);
            fflush(stdout); // Force output to appear immediately
        }
        else
        {
            Tester->Mode = TEST_MODE_COMPLETED;
            fprintf(stdout, "\n"); // New line when completed
        }
    }

    return Tester->Mode == TEST_MODE_TESTING;
}
