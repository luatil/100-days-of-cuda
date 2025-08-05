#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>

#define CUPTI_CALL(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        CUptiResult _status = call;                                                                                    \
        if (_status != CUPTI_SUCCESS)                                                                                  \
        {                                                                                                              \
            const char *errstr;                                                                                        \
            cuptiGetResultString(_status, &errstr);                                                                    \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr);   \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

#define DRIVER_API_CALL(apiFuncCall)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult _status = apiFuncCall;                                                                                \
        if (_status != CUDA_SUCCESS)                                                                                   \
        {                                                                                                              \
            fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", __FILE__, __LINE__, #apiFuncCall,     \
                    _status);                                                                                          \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

int main()
{
    CUdevice Device;
    int DeviceCount;
    char DeviceName[256];

    // Initialize CUDA
    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGetCount(&DeviceCount));

    if (DeviceCount == 0)
    {
        printf("Error: No CUDA devices found\n");
        return -1;
    }

    // Use first device
    DRIVER_API_CALL(cuDeviceGet(&Device, 0));
    DRIVER_API_CALL(cuDeviceGetName(DeviceName, sizeof(DeviceName), Device));
    printf("CUDA Device: %s\n\n", DeviceName);

    // Create context
    CUcontext Context;
    DRIVER_API_CALL(cuCtxCreate(&Context, 0, Device));

    // Get number of metrics
    uint32_t NumMetrics;
    CUPTI_CALL(cuptiDeviceGetNumMetrics(Device, &NumMetrics));
    printf("Total metrics available: %d\n\n", NumMetrics);

    // Allocate space for metric IDs
    CUpti_MetricID *MetricIds = (CUpti_MetricID *)malloc(NumMetrics * sizeof(CUpti_MetricID));

    // Get all metric IDs
    size_t ArraySize = NumMetrics;
    CUPTI_CALL(cuptiDeviceEnumMetrics(Device, &ArraySize, MetricIds));

    printf("Available metrics:\n");
    printf("==================\n");

    // Print each metric name and description
    for (uint32_t I = 0; I < NumMetrics; I++)
    {
        char MetricName[256];
        char ShortDescription[512];
        char LongDescription[1024];
        CUpti_MetricCategory Category;
        CUpti_MetricEvaluationMode EvalMode;
        CUpti_MetricValueKind ValueKind;

        size_t Size;

        // Get metric name
        Size = sizeof(MetricName);
        CUPTI_CALL(cuptiMetricGetAttribute(MetricIds[I], CUPTI_METRIC_ATTR_NAME, &Size, MetricName));

        // Get short description
        Size = sizeof(ShortDescription);
        CUPTI_CALL(cuptiMetricGetAttribute(MetricIds[I], CUPTI_METRIC_ATTR_SHORT_DESCRIPTION, &Size, ShortDescription));

        // Get category
        Size = sizeof(Category);
        CUPTI_CALL(cuptiMetricGetAttribute(MetricIds[I], CUPTI_METRIC_ATTR_CATEGORY, &Size, &Category));

        // Get evaluation mode
        Size = sizeof(EvalMode);
        CUPTI_CALL(cuptiMetricGetAttribute(MetricIds[I], CUPTI_METRIC_ATTR_EVALUATION_MODE, &Size, &EvalMode));

        // Get value kind
        Size = sizeof(ValueKind);
        CUPTI_CALL(cuptiMetricGetAttribute(MetricIds[I], CUPTI_METRIC_ATTR_VALUE_KIND, &Size, &ValueKind));

        printf("%3d. %-30s - %s\n", I + 1, MetricName, ShortDescription);

        const char *CategoryStr = "Unknown";
        switch (Category)
        {
        case CUPTI_METRIC_CATEGORY_MEMORY:
            CategoryStr = "Memory";
            break;
        case CUPTI_METRIC_CATEGORY_INSTRUCTION:
            CategoryStr = "Instruction";
            break;
        case CUPTI_METRIC_CATEGORY_MULTIPROCESSOR:
            CategoryStr = "Multiprocessor";
            break;
        case CUPTI_METRIC_CATEGORY_CACHE:
            CategoryStr = "Cache";
            break;
        case CUPTI_METRIC_CATEGORY_TEXTURE:
            CategoryStr = "Texture";
            break;
        }

        const char *EvalModeStr = "Unknown";
        switch (EvalMode)
        {
        case CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE:
            EvalModeStr = "Per Instance";
            break;
        case CUPTI_METRIC_EVALUATION_MODE_AGGREGATE:
            EvalModeStr = "Aggregate";
            break;
        }

        const char *ValueKindStr = "Unknown";
        switch (ValueKind)
        {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE:
            ValueKindStr = "Double";
            break;
        case CUPTI_METRIC_VALUE_KIND_UINT64:
            ValueKindStr = "Uint64";
            break;
        case CUPTI_METRIC_VALUE_KIND_PERCENT:
            ValueKindStr = "Percent";
            break;
        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
            ValueKindStr = "Throughput";
            break;
        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
            ValueKindStr = "Utilization Level";
            break;
        }

        printf("     Category: %s, Eval Mode: %s, Value Kind: %s\n\n", CategoryStr, EvalModeStr, ValueKindStr);
    }

    // Cleanup
    free(MetricIds);
    DRIVER_API_CALL(cuCtxDestroy(Context));

    return 0;
}