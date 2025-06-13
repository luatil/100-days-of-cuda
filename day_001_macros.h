#ifndef DAY_001_MACROS_H
#define DAY_001_MACROS_H
#include <stdio.h>

#define AllocateCPU(_Type, _NumberOfElements) (_Type *)malloc(sizeof(_Type) * _NumberOfElements)

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

#endif // DAY_001_MACROS_H
