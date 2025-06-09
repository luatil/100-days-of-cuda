/*
 * Day 07: Querying device properties
 *
 * Based on chapter 4 from PMPP.
 *
 */
#include <cuda_runtime.h>
#include <stdlib.h>

typedef float f32;
typedef unsigned char u8;
typedef unsigned int u32;
typedef int s32;

#include "day_001_macros.h"

int main()
{
    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);

    DbgS32(DeviceCount);

    cudaDeviceProp DeviceProperties;
    for (int CudaDevice = 0; CudaDevice < DeviceCount; CudaDevice++)
    {
        cudaGetDeviceProperties(&DeviceProperties, CudaDevice);
        DbgU32(CudaDevice);

        DbgS32(DeviceProperties.maxThreadsPerBlock);
        DbgS32(DeviceProperties.multiProcessorCount);
        DbgS32(DeviceProperties.clockRate);

        DbgS32(DeviceProperties.maxThreadsDim[0]);
        DbgS32(DeviceProperties.maxThreadsDim[1]);
        DbgS32(DeviceProperties.maxThreadsDim[2]);

        DbgS32(DeviceProperties.regsPerBlock);
        DbgS32(DeviceProperties.regsPerMultiprocessor);
        DbgS32(DeviceProperties.warpSize);
    }
}
