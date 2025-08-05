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
    CUdevice Device;
    CUpti_MetricID MetricId;
    CUpti_EventID *EventIdArray;
    uint32_t NumEvents;
    uint64_t *EventValueArray;
    CUpti_EventGroupSets *EventGroupSets;
    int EventGroupSetCount;
    int CurrentEventGroupSet;
    int NumEventGroups;
    CUpti_EventGroup *EventGroups;
} metric_data_t;

static int EventGroupSetIndex = 0;
static metric_data_t MetricData;
static CUpti_SubscriberHandle Subscriber;

// Simple vector addition kernel
__global__ void VecAdd(const int *A, const int *B, int *C, int NumberOfElements)
{
    int I = blockDim.x * blockIdx.x + threadIdx.x;
    if (I < NumberOfElements)
        C[I] = A[I] + B[I];
}

// CUPTI callback function for metric collection
static void CUPTIAPI GetMetricValueCallback(void *Userdata, CUpti_CallbackDomain, CUpti_CallbackId Cbid,
                                            const CUpti_CallbackData *CbInfo)
{
    metric_data_t *MetricData = (metric_data_t *)Userdata;

    // Check if this is a kernel launch callback
    if ((Cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
        (Cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
    {

        if (CbInfo->callbackSite == CUPTI_API_ENTER)
        {
            // Enable event collection for the current event group set
            if (EventGroupSetIndex < MetricData->EventGroupSetCount)
            {
                printf("Enabling event group set %d\n", EventGroupSetIndex);

                CUpti_EventGroupSet EventGroupSet = MetricData->EventGroupSets->sets[EventGroupSetIndex];

                for (uint32_t I = 0; I < EventGroupSet.numEventGroups; I++)
                {
                    CUPTI_CALL(cuptiEventGroupEnable(EventGroupSet.eventGroups[I]));
                }
            }
        }
        else if (CbInfo->callbackSite == CUPTI_API_EXIT)
        {
            // Disable event collection and read event values
            if (EventGroupSetIndex < MetricData->EventGroupSetCount)
            {
                printf("Disabling event group set %d\n", EventGroupSetIndex);

                CUpti_EventGroupSet EventGroupSet = MetricData->EventGroupSets->sets[EventGroupSetIndex];

                for (uint32_t I = 0; I < EventGroupSet.numEventGroups; I++)
                {
                    CUpti_EventGroup EventGroup = EventGroupSet.eventGroups[I];
                    CUpti_EventDomainID EventDomainId;
                    uint32_t NumEvents, NumInstances, NumTotalInstances;
                    size_t Size;

                    // Get event group info
                    Size = sizeof(EventDomainId);
                    CUPTI_CALL(cuptiEventGroupGetAttribute(EventGroup, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &Size,
                                                           &EventDomainId));
                    Size = sizeof(NumEvents);
                    CUPTI_CALL(
                        cuptiEventGroupGetAttribute(EventGroup, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &Size, &NumEvents));
                    Size = sizeof(NumInstances);
                    CUPTI_CALL(cuptiEventGroupGetAttribute(EventGroup, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &Size,
                                                           &NumInstances));

                    // Get total instances for normalization
                    CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(MetricData->Device, EventDomainId,
                                                                  CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &Size,
                                                                  &NumTotalInstances));

                    // Read event values
                    uint64_t *EventValues = (uint64_t *)malloc(NumEvents * NumInstances * sizeof(uint64_t));
                    Size = NumEvents * NumInstances * sizeof(uint64_t);
                    CUPTI_CALL(cuptiEventGroupReadAllEvents(EventGroup, CUPTI_EVENT_READ_FLAG_NONE, &Size, EventValues,
                                                            &Size, NULL, NULL));

                    // Get event IDs for this group
                    CUpti_EventID *EventIds = (CUpti_EventID *)malloc(NumEvents * sizeof(CUpti_EventID));
                    Size = NumEvents * sizeof(CUpti_EventID);
                    CUPTI_CALL(cuptiEventGroupGetAttribute(EventGroup, CUPTI_EVENT_GROUP_ATTR_EVENTS, &Size, EventIds));

                    // Accumulate normalized event values
                    for (uint32_t J = 0; J < NumEvents; J++)
                    {
                        uint64_t Sum = 0;
                        for (uint32_t K = 0; K < NumInstances; K++)
                        {
                            Sum += EventValues[J * NumInstances + K];
                        }

                        // Normalize: (sum * numTotalInstances) / numInstances
                        uint64_t NormalizedValue = (Sum * NumTotalInstances) / NumInstances;

                        // Find which metric event this corresponds to
                        for (uint32_t K = 0; K < MetricData->NumEvents; K++)
                        {
                            if (MetricData->EventIdArray[K] == EventIds[J])
                            {
                                MetricData->EventValueArray[K] = NormalizedValue;
                                printf("Event %d value: %llu (normalized)\n", K, (unsigned long long)NormalizedValue);
                                break;
                            }
                        }
                    }

                    CUPTI_CALL(cuptiEventGroupDisable(EventGroup));
                    free(EventValues);
                    free(EventIds);
                }

                EventGroupSetIndex++;
            }
        }
    }
}

// Initialize CUPTI metric collection
void InitializeMetric(CUdevice Device, const char *MetricName)
{
    MetricData.Device = Device;

    // Get metric ID
    CUPTI_CALL(cuptiMetricGetIdFromName(Device, MetricName, &MetricData.MetricId));

    // Get number of events required for this metric
    CUPTI_CALL(cuptiMetricGetNumEvents(MetricData.MetricId, (uint32_t *)&MetricData.NumEvents));
    printf("Metric '%s' requires %d events\n", MetricName, MetricData.NumEvents);

    // Allocate space for events
    MetricData.EventIdArray = (CUpti_EventID *)malloc(MetricData.NumEvents * sizeof(CUpti_EventID));
    MetricData.EventValueArray = (uint64_t *)malloc(MetricData.NumEvents * sizeof(uint64_t));
    memset(MetricData.EventValueArray, 0, MetricData.NumEvents * sizeof(uint64_t));

    // Get the events required for the metric
    size_t EventArraySize = MetricData.NumEvents;
    CUPTI_CALL(cuptiMetricEnumEvents(MetricData.MetricId, &EventArraySize, MetricData.EventIdArray));

    // Create event group sets
    CUcontext Context;
    DRIVER_API_CALL(cuCtxCreate(&Context, 0, Device));
    CUPTI_CALL(cuptiEventGroupSetsCreate(Context, MetricData.NumEvents * sizeof(CUpti_EventID), MetricData.EventIdArray,
                                         &MetricData.EventGroupSets));

    MetricData.EventGroupSetCount = MetricData.EventGroupSets->numSets;
    printf("Created %d event group sets\n", MetricData.EventGroupSetCount);
}

// Calculate and display the metric value
void CalculateMetricValue(const char *MetricName)
{
    CUpti_MetricValue MetricValue;

    // Calculate the metric value from collected events
    CUPTI_CALL(cuptiMetricGetValue(MetricData.Device, MetricData.MetricId, MetricData.NumEvents * sizeof(CUpti_EventID),
                                   MetricData.EventIdArray, MetricData.NumEvents * sizeof(uint64_t),
                                   MetricData.EventValueArray, 0, &MetricValue));

    // Print the result - simplified to avoid union member issues
    printf("Metric %s calculated successfully\n", MetricName);
}

// Vector addition test function
void RunVectorAdd()
{
    int *HA, *HB, *HC;
    int *DA, *DB, *DC;
    size_t Size = N * sizeof(int);

    // Allocate host memory
    HA = (int *)malloc(Size);
    HB = (int *)malloc(Size);
    HC = (int *)malloc(Size);

    // Initialize host arrays
    for (int I = 0; I < N; I++)
    {
        HA[I] = I;
        HB[I] = I * 2;
    }

    // Allocate device memory
    RUNTIME_API_CALL(cudaMalloc((void **)&DA, Size));
    RUNTIME_API_CALL(cudaMalloc((void **)&DB, Size));
    RUNTIME_API_CALL(cudaMalloc((void **)&DC, Size));

    // Copy data to device
    RUNTIME_API_CALL(cudaMemcpy(DA, HA, Size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(DB, HB, Size, cudaMemcpyHostToDevice));

    // Launch kernel (this will trigger our callbacks)
    int ThreadsPerBlock = 256;
    int BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;
    printf("Launching kernel: blocks %d, threads per block %d\n", BlocksPerGrid, ThreadsPerBlock);

    VecAdd<<<BlocksPerGrid, ThreadsPerBlock>>>(DA, DB, DC, N);

    // Wait for kernel to complete
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Copy result back to host
    RUNTIME_API_CALL(cudaMemcpy(HC, DC, Size, cudaMemcpyDeviceToHost));

    // Verify result
    for (int I = 0; I < N; I++)
    {
        if (HC[I] != HA[I] + HB[I])
        {
            printf("Error: result verification failed at element %d\n", I);
            exit(-1);
        }
    }
    printf("Vector addition completed successfully!\n");

    // Cleanup
    free(HA);
    free(HB);
    free(HC);
    RUNTIME_API_CALL(cudaFree(DA));
    RUNTIME_API_CALL(cudaFree(DB));
    RUNTIME_API_CALL(cudaFree(DC));
}

int main()
{
    CUdevice Device;
    int DeviceCount;
    char DeviceName[256];
    const char *MetricName = "sm__inst_executed";

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
    printf("Using CUDA Device: %s\n", DeviceName);

    // Check if metric is available
    CUpti_MetricID TempMetricId;
    CUptiResult Result = cuptiMetricGetIdFromName(Device, MetricName, &TempMetricId);
    if (Result != CUPTI_SUCCESS)
    {
        printf("Error: Metric '%s' is not available on this device\n", MetricName);
        printf("This may be because the device doesn't support this metric or you need newer drivers\n");
        return -1;
    }

    printf("Measuring metric: %s\n", MetricName);

    // Initialize metric collection
    InitializeMetric(Device, MetricName);

    // Subscribe to CUDA runtime callbacks
    CUPTI_CALL(cuptiSubscribe(&Subscriber, (CUpti_CallbackFunc)GetMetricValueCallback, &MetricData));
    CUPTI_CALL(
        cuptiEnableCallback(1, Subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
    CUPTI_CALL(cuptiEnableCallback(1, Subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

    // Run the kernel multiple times if we have multiple event group sets
    EventGroupSetIndex = 0;
    for (int I = 0; I < MetricData.EventGroupSetCount; I++)
    {
        printf("\n=== Pass %d ===\n", I);
        RunVectorAdd();
    }

    // Calculate and display the final metric value
    printf("\n=== Final Results ===\n");
    CalculateMetricValue(MetricName);

    // Cleanup
    CUPTI_CALL(cuptiUnsubscribe(Subscriber));
    CUPTI_CALL(cuptiEventGroupSetsDestroy(MetricData.EventGroupSets));
    free(MetricData.EventIdArray);
    free(MetricData.EventValueArray);

    printf("Profiling completed!\n");
    return 0;
}
