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
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr); \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

#define DRIVER_API_CALL(apiFuncCall)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult _status = apiFuncCall;                                                                                \
        if (_status != CUDA_SUCCESS)                                                                                   \
        {                                                                                                              \
            fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", __FILE__, __LINE__, #apiFuncCall,   \
                    _status);                                                                                          \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

int main()
{
    CUdevice device;
    int deviceCount;
    char deviceName[256];
    
    // Initialize CUDA
    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    
    if (deviceCount == 0)
    {
        printf("Error: No CUDA devices found\n");
        return -1;
    }
    
    // Use first device
    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, sizeof(deviceName), device));
    printf("CUDA Device: %s\n\n", deviceName);
    
    // Create context
    CUcontext context;
    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
    
    // Get number of metrics
    uint32_t numMetrics;
    CUPTI_CALL(cuptiDeviceGetNumMetrics(device, &numMetrics));
    printf("Total metrics available: %d\n\n", numMetrics);
    
    // Allocate space for metric IDs
    CUpti_MetricID *metricIds = (CUpti_MetricID *)malloc(numMetrics * sizeof(CUpti_MetricID));
    
    // Get all metric IDs
    size_t arraySize = numMetrics;
    CUPTI_CALL(cuptiDeviceEnumMetrics(device, &arraySize, metricIds));
    
    printf("Available metrics:\n");
    printf("==================\n");
    
    // Print each metric name and description
    for (uint32_t i = 0; i < numMetrics; i++)
    {
        char metricName[256];
        char shortDescription[512];
        char longDescription[1024];
        CUpti_MetricCategory category;
        CUpti_MetricEvaluationMode evalMode;
        CUpti_MetricValueKind valueKind;
        
        size_t size;
        
        // Get metric name
        size = sizeof(metricName);
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], CUPTI_METRIC_ATTR_NAME, &size, metricName));
        
        // Get short description
        size = sizeof(shortDescription);
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], CUPTI_METRIC_ATTR_SHORT_DESCRIPTION, &size, shortDescription));
        
        // Get category
        size = sizeof(category);
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], CUPTI_METRIC_ATTR_CATEGORY, &size, &category));
        
        // Get evaluation mode
        size = sizeof(evalMode);
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], CUPTI_METRIC_ATTR_EVALUATION_MODE, &size, &evalMode));
        
        // Get value kind
        size = sizeof(valueKind);
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], CUPTI_METRIC_ATTR_VALUE_KIND, &size, &valueKind));
        
        printf("%3d. %-30s - %s\n", i+1, metricName, shortDescription);
        
        const char *categoryStr = "Unknown";
        switch (category) {
            case CUPTI_METRIC_CATEGORY_MEMORY: categoryStr = "Memory"; break;
            case CUPTI_METRIC_CATEGORY_INSTRUCTION: categoryStr = "Instruction"; break;
            case CUPTI_METRIC_CATEGORY_MULTIPROCESSOR: categoryStr = "Multiprocessor"; break;
            case CUPTI_METRIC_CATEGORY_CACHE: categoryStr = "Cache"; break;
            case CUPTI_METRIC_CATEGORY_TEXTURE: categoryStr = "Texture"; break;
        }
        
        const char *evalModeStr = "Unknown";
        switch (evalMode) {
            case CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE: evalModeStr = "Per Instance"; break;
            case CUPTI_METRIC_EVALUATION_MODE_AGGREGATE: evalModeStr = "Aggregate"; break;
        }
        
        const char *valueKindStr = "Unknown";
        switch (valueKind) {
            case CUPTI_METRIC_VALUE_KIND_DOUBLE: valueKindStr = "Double"; break;
            case CUPTI_METRIC_VALUE_KIND_UINT64: valueKindStr = "Uint64"; break;
            case CUPTI_METRIC_VALUE_KIND_PERCENT: valueKindStr = "Percent"; break;
            case CUPTI_METRIC_VALUE_KIND_THROUGHPUT: valueKindStr = "Throughput"; break;
            case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL: valueKindStr = "Utilization Level"; break;
        }
        
        printf("     Category: %s, Eval Mode: %s, Value Kind: %s\n\n", categoryStr, evalModeStr, valueKindStr);
    }
    
    // Cleanup
    free(metricIds);
    DRIVER_API_CALL(cuCtxDestroy(context));
    
    return 0;
}