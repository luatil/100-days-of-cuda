#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CU(call)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult _err = (call);                                                                                        \
        if (_err != CUDA_SUCCESS)                                                                                      \
        {                                                                                                              \
            const char *_str;                                                                                          \
            cuGetErrorString(_err, &_str);                                                                             \
            fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__, __LINE__, _str);                             \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define N (1 << 20) // 1M elements
#define THREADS 256

int main()
{
    CHECK_CU(cuInit(0));

    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    // cuCtxCreate_v4 (CUDA 13+) requires a params struct; pass NULL for defaults.
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, NULL, 0, device));

    // Load PTX from disk, passing a JIT error log buffer so compilation
    // failures print the actual ptxas error instead of a generic code.
    char ptx_error_log[8192] = {0};
    CUjit_option jit_opts[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void *jit_vals[] = {ptx_error_log, (void *)(uintptr_t)sizeof(ptx_error_log)};

    // FILE *ptx_file = fopen("day_084_kernel.ptx", "rb");
    FILE *ptx_file = fopen("day_084_nvcc_vector_add.ptx", "rb");
    fseek(ptx_file, 0, SEEK_END);
    long ptx_size = ftell(ptx_file);
    rewind(ptx_file);
    char *ptx_src = (char *)malloc(ptx_size + 1);
    fread(ptx_src, 1, ptx_size, ptx_file);
    ptx_src[ptx_size] = '\0';
    fclose(ptx_file);

    CUmodule module;
    CUresult load_err = cuModuleLoadDataEx(&module, ptx_src, 2, jit_opts, jit_vals);
    free(ptx_src);
    if (load_err != CUDA_SUCCESS)
    {
        fprintf(stderr, "PTX JIT error:\n%s\n", ptx_error_log);
        exit(1);
    }

    CUfunction kernel;
    // CHECK_CU(cuModuleGetFunction(&kernel, module, "vector_add"));
    CHECK_CU(cuModuleGetFunction(&kernel, module, "VectorAdd"));

    // ---- host arrays ----
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // ---- device arrays ----
    CUdeviceptr d_a, d_b, d_c;
    CHECK_CU(cuMemAlloc(&d_a, N * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_b, N * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_c, N * sizeof(float)));

    CHECK_CU(cuMemcpyHtoD(d_a, h_a, N * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_b, h_b, N * sizeof(float)));

    // ---- launch ----
    int n = N;
    int blocks = (N + THREADS - 1) / THREADS;
    void *args[] = {&d_a, &d_b, &d_c, &n};
    CHECK_CU(cuLaunchKernel(kernel, blocks, 1, 1, // gridDim
                            THREADS, 1, 1,        // blockDim
                            0, NULL,              // sharedMem, stream
                            args, NULL));

    CHECK_CU(cuCtxSynchronize());

    // ---- copy back and verify ----
    CHECK_CU(cuMemcpyDtoH(h_c, d_c, N * sizeof(float)));

    int errors = 0;
    for (int i = 0; i < N; i++)
    {
        float expected = (float)N; // i + (N - i) = N for every element
        if (h_c[i] != expected)
        {
            if (errors < 5)
                fprintf(stderr, "FAIL at %d: got %f, expected %f\n", i, h_c[i], expected);
            errors++;
        }
    }
    if (errors == 0)
        printf("PASS: %d elements, each = %.1f\n", N, (float)N);
    else
        printf("FAIL: %d errors\n", errors);

    // ---- cleanup ----
    CHECK_CU(cuMemFree(d_a));
    CHECK_CU(cuMemFree(d_b));
    CHECK_CU(cuMemFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CU(cuModuleUnload(module));
    CHECK_CU(cuCtxDestroy(ctx));

    return errors != 0;
}
