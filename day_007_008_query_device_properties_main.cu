/*
 * Day 07: Querying device properties
 *
 * Based on chapter 4 from PMPP.
 *
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

typedef float f32;
typedef unsigned char u8;
typedef unsigned int u32;
typedef int s32;

#define AllocateCPU(_Type, _NumberOfElements) ((_Type *)malloc(sizeof(_Type) * (_NumberOfElements)))

#ifndef DEBUG_ENABLED
#define DEBUG_ENABLED 1
#endif

#if DEBUG_ENABLED
#define DbgU32(_Val) printf(#_Val "=%d\n", (_Val))
#define DbgS32(_Val) printf(#_Val "=%d\n", (_Val))
#define DbgF32(_Val) printf(#_Val "=%f\n", (_Val))
#else
#define DbgU32(_Val)
#define DbgS32(_Val)
#define DbgF32(_Val)
#endif

int main()
{
    s32 DeviceCount;
    cudaGetDeviceCount(&DeviceCount);

    DbgS32(DeviceCount);

    cudaDeviceProp DeviceProperties;
    for (u32 CudaDevice = 0; CudaDevice < DeviceCount; CudaDevice++)
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
