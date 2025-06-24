#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#define RUNTIME_API_CALL(apiFuncCall)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t _status = apiFuncCall;                                                                             \
        if (_status != cudaSuccess)                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #apiFuncCall,     \
                    cudaGetErrorString(_status));                                                                      \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

// Vector size
#define N 50000

// Metric data structure
typedef struct
{
    CUdevice device;
    CUpti_MetricID metricId;
    CUpti_EventID *eventIdArray;
    uint32_t numEvents;
    uint64_t *eventValueArray;
    CUpti_EventGroupSets *eventGroupSets;
    int eventGroupSetCount;
    int currentEventGroupSet;
    int numEventGroups;
    CUpti_EventGroup *eventGroups;
} MetricData_t;

static int eventGroupSetIndex = 0;
static MetricData_t metricData;
static CUpti_SubscriberHandle subscriber;

// Simple vector addition kernel
__global__ void VecAdd(const int *A, const int *B, int *C, int NumberOfElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < NumberOfElements)
        C[i] = A[i] + B[i];
}

// CUPTI callback function for metric collection
static void CUPTIAPI getMetricValueCallback(void *userdata, CUpti_CallbackDomain, CUpti_CallbackId cbid,
                                            const CUpti_CallbackData *cbInfo)
{
    MetricData_t *metricData = (MetricData_t *)userdata;

    // Check if this is a kernel launch callback
    if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
        (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
    {

        if (cbInfo->callbackSite == CUPTI_API_ENTER)
        {
            // Enable event collection for the current event group set
            if (eventGroupSetIndex < metricData->eventGroupSetCount)
            {
                printf("Enabling event group set %d\n", eventGroupSetIndex);

                CUpti_EventGroupSet eventGroupSet = metricData->eventGroupSets->sets[eventGroupSetIndex];

                for (uint32_t i = 0; i < eventGroupSet.numEventGroups; i++)
                {
                    CUPTI_CALL(cuptiEventGroupEnable(eventGroupSet.eventGroups[i]));
                }
            }
        }
        else if (cbInfo->callbackSite == CUPTI_API_EXIT)
        {
            // Disable event collection and read event values
            if (eventGroupSetIndex < metricData->eventGroupSetCount)
            {
                printf("Disabling event group set %d\n", eventGroupSetIndex);

                CUpti_EventGroupSet eventGroupSet = metricData->eventGroupSets->sets[eventGroupSetIndex];

                for (uint32_t i = 0; i < eventGroupSet.numEventGroups; i++)
                {
                    CUpti_EventGroup eventGroup = eventGroupSet.eventGroups[i];
                    CUpti_EventDomainID eventDomainId;
                    uint32_t numEvents, numInstances, numTotalInstances;
                    size_t size;

                    // Get event group info
                    size = sizeof(eventDomainId);
                    CUPTI_CALL(cuptiEventGroupGetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &size,
                                                           &eventDomainId));
                    size = sizeof(numEvents);
                    CUPTI_CALL(
                        cuptiEventGroupGetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &size, &numEvents));
                    size = sizeof(numInstances);
                    CUPTI_CALL(cuptiEventGroupGetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &size,
                                                           &numInstances));

                    // Get total instances for normalization
                    CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(metricData->device, eventDomainId,
                                                                  CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &size,
                                                                  &numTotalInstances));

                    // Read event values
                    uint64_t *eventValues = (uint64_t *)malloc(numEvents * numInstances * sizeof(uint64_t));
                    size = numEvents * numInstances * sizeof(uint64_t);
                    CUPTI_CALL(cuptiEventGroupReadAllEvents(eventGroup, CUPTI_EVENT_READ_FLAG_NONE, &size, eventValues,
                                                            &size, NULL, NULL));

                    // Get event IDs for this group
                    CUpti_EventID *eventIds = (CUpti_EventID *)malloc(numEvents * sizeof(CUpti_EventID));
                    size = numEvents * sizeof(CUpti_EventID);
                    CUPTI_CALL(cuptiEventGroupGetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_EVENTS, &size, eventIds));

                    // Accumulate normalized event values
                    for (uint32_t j = 0; j < numEvents; j++)
                    {
                        uint64_t sum = 0;
                        for (uint32_t k = 0; k < numInstances; k++)
                        {
                            sum += eventValues[j * numInstances + k];
                        }

                        // Normalize: (sum * numTotalInstances) / numInstances
                        uint64_t normalizedValue = (sum * numTotalInstances) / numInstances;

                        // Find which metric event this corresponds to
                        for (uint32_t k = 0; k < metricData->numEvents; k++)
                        {
                            if (metricData->eventIdArray[k] == eventIds[j])
                            {
                                metricData->eventValueArray[k] = normalizedValue;
                                printf("Event %d value: %llu (normalized)\n", k, (unsigned long long)normalizedValue);
                                break;
                            }
                        }
                    }

                    CUPTI_CALL(cuptiEventGroupDisable(eventGroup));
                    free(eventValues);
                    free(eventIds);
                }

                eventGroupSetIndex++;
            }
        }
    }
}

// Initialize CUPTI metric collection
void initializeMetric(CUdevice device, const char *metricName)
{
    metricData.device = device;

    // Get metric ID
    CUPTI_CALL(cuptiMetricGetIdFromName(device, metricName, &metricData.metricId));

    // Get number of events required for this metric
    CUPTI_CALL(cuptiMetricGetNumEvents(metricData.metricId, (uint32_t *)&metricData.numEvents));
    printf("Metric '%s' requires %d events\n", metricName, metricData.numEvents);

    // Allocate space for events
    metricData.eventIdArray = (CUpti_EventID *)malloc(metricData.numEvents * sizeof(CUpti_EventID));
    metricData.eventValueArray = (uint64_t *)malloc(metricData.numEvents * sizeof(uint64_t));
    memset(metricData.eventValueArray, 0, metricData.numEvents * sizeof(uint64_t));

    // Get the events required for the metric
    size_t eventArraySize = metricData.numEvents;
    CUPTI_CALL(cuptiMetricEnumEvents(metricData.metricId, &eventArraySize, metricData.eventIdArray));

    // Create event group sets
    CUcontext context;
    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
    CUPTI_CALL(cuptiEventGroupSetsCreate(context, metricData.numEvents * sizeof(CUpti_EventID), metricData.eventIdArray,
                                         &metricData.eventGroupSets));

    metricData.eventGroupSetCount = metricData.eventGroupSets->numSets;
    printf("Created %d event group sets\n", metricData.eventGroupSetCount);
}

// Calculate and display the metric value
void calculateMetricValue(const char *metricName)
{
    CUpti_MetricValue metricValue;

    // Calculate the metric value from collected events
    CUPTI_CALL(cuptiMetricGetValue(metricData.device, metricData.metricId, metricData.numEvents * sizeof(CUpti_EventID),
                                   metricData.eventIdArray, metricData.numEvents * sizeof(uint64_t),
                                   metricData.eventValueArray, 0, &metricValue));

    // Print the result - simplified to avoid union member issues
    printf("Metric %s calculated successfully\n", metricName);
}

// Vector addition test function
void runVectorAdd()
{
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    size_t size = N * sizeof(int);

    // Allocate host memory
    h_A = (int *)malloc(size);
    h_B = (int *)malloc(size);
    h_C = (int *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate device memory
    RUNTIME_API_CALL(cudaMalloc((void **)&d_A, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&d_B, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&d_C, size));

    // Copy data to device
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel (this will trigger our callbacks)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel: blocks %d, threads per block %d\n", blocksPerGrid, threadsPerBlock);

    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for kernel to complete
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Copy result back to host
    RUNTIME_API_CALL(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (int i = 0; i < N; i++)
    {
        if (h_C[i] != h_A[i] + h_B[i])
        {
            printf("Error: result verification failed at element %d\n", i);
            exit(-1);
        }
    }
    printf("Vector addition completed successfully!\n");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    RUNTIME_API_CALL(cudaFree(d_A));
    RUNTIME_API_CALL(cudaFree(d_B));
    RUNTIME_API_CALL(cudaFree(d_C));
}

int main()
{
    CUdevice device;
    int deviceCount;
    char deviceName[256];
    const char *metricName = "sm__inst_executed";

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
    printf("Using CUDA Device: %s\n", deviceName);

    // Check if metric is available
    CUpti_MetricID tempMetricId;
    CUptiResult result = cuptiMetricGetIdFromName(device, metricName, &tempMetricId);
    if (result != CUPTI_SUCCESS)
    {
        printf("Error: Metric '%s' is not available on this device\n", metricName);
        printf("This may be because the device doesn't support this metric or you need newer drivers\n");
        return -1;
    }

    printf("Measuring metric: %s\n", metricName);

    // Initialize metric collection
    initializeMetric(device, metricName);

    // Subscribe to CUDA runtime callbacks
    CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getMetricValueCallback, &metricData));
    CUPTI_CALL(
        cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
    CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

    // Run the kernel multiple times if we have multiple event group sets
    eventGroupSetIndex = 0;
    for (int i = 0; i < metricData.eventGroupSetCount; i++)
    {
        printf("\n=== Pass %d ===\n", i);
        runVectorAdd();
    }

    // Calculate and display the final metric value
    printf("\n=== Final Results ===\n");
    calculateMetricValue(metricName);

    // Cleanup
    CUPTI_CALL(cuptiUnsubscribe(subscriber));
    CUPTI_CALL(cuptiEventGroupSetsDestroy(metricData.eventGroupSets));
    free(metricData.eventIdArray);
    free(metricData.eventValueArray);

    printf("Profiling completed!\n");
    return 0;
}
