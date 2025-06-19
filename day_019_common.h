#ifndef DAY_019_COMMON_H
#define DAY_019_COMMON_H
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;

#define AllocateCPU(_Type, _NumberOfElements) (_Type *)malloc(sizeof(_Type) * _NumberOfElements)
#define FreeCPU(_Ptr) free(_Ptr)

#ifndef Assert
#define Assert(_Expr) assert(_Expr)
#endif

#ifndef DEBUG_ENABLED
#define DEBUG_ENABLED 1
#endif

#if DEBUG_ENABLED
#define DbgU32(_Val) printf(#_Val "=%d\n", (_Val))
#define DbgU64(_Val) printf(#_Val "=%ld\n", (_Val))
#define DbgF32(_Val) printf(#_Val "=%f\n", (_Val))
#define DbgS32(_Val) printf(#_Val "=%d\n", (_Val))
#else
#define DbgU32(_Val)
#define DbgU64(_Val)
#define DbgF32(_Val)
#define DbgS32(_Val)
#endif

#if DEBUG_ENABLED
#define Dbg(_Val) printf(#_Val "=%.4f\n", (double)(_Val))
#else
#define Dbg(_Val)
#endif

#if DEBUG_ENABLED
#define DbgCudaF32(_Val)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        f32 *temp_host = AllocateCPU(f32, 1);                                                                          \
        cudaMemcpy(temp_host, &(_Val), sizeof(_Val), cudaMemcpyDeviceToHost);                                          \
        printf(#_Val "=%.4f\n", *temp_host);                                                                           \
        FreeCPU(temp_host);                                                                                            \
    } while (0)
#else
#define DbgCuda(_Val)
#endif

#endif // DAY_019_COMMON_H
