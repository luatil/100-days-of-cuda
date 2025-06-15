#ifndef DAY_015_COMMON_H
#define DAY_015_COMMON_H
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

#endif // DAY_015_COMMON_H
