#include <stdio.h>
#include <cuda_runtime.h>

#include "day_066_munit_include.h"

__device__ int add_device(int a, int b) {
    return a + b;
}

__global__ void add_kernel(int a, int b, int *result) {
    *result = add_device(a, b);
}

int add_host(int a, int b) {
    return a + b;
}

static MunitResult test_host_addition(const MunitParameter params[], void* data) {
    (void)params;
    (void)data;
    munit_assert_int(add_host(2, 3), ==, 5);
    munit_assert_int(add_host(-1, 1), ==, 0);
    munit_assert_int(add_host(0, 0), ==, 0);
    return MUNIT_OK;
}

static MunitResult test_cuda_addition(const MunitParameter params[], void* data) {
    (void) params;
    (void) data;
    int *d_result;
    int h_result;
    
    cudaMalloc(&d_result, sizeof(int));
    
    add_kernel<<<1, 1>>>(2, 3, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    munit_assert_int(h_result, ==, 5);
    
    add_kernel<<<1, 1>>>(-10, 15, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    munit_assert_int(h_result, ==, 5);
    
    cudaFree(d_result);
    return MUNIT_OK;
}

static MunitResult test_string_operations(const MunitParameter params[], void* data) {
    (void)params;
    (void)data;
    const char* str1 = "hello";
    const char* str2 = "hello";
    const char* str3 = "world";
    
    munit_assert_string_equal(str1, str2);
    munit_assert_string_not_equal(str1, str3);
    
    return MUNIT_OK;
}

static MunitResult test_memory_operations(const MunitParameter params[], void* data) {
    (void)params;
    (void)data;
    unsigned char buf1[] = {1, 2, 3, 4};
    unsigned char buf2[] = {1, 2, 3, 4};
    unsigned char buf3[] = {1, 2, 3, 5};
    
    munit_assert_memory_equal(4, buf1, buf2);
    munit_assert_memory_not_equal(4, buf1, buf3);
    
    return MUNIT_OK;
}

static MunitResult test_floating_point(const MunitParameter params[], void* data) {
    (void)params;
    (void)data;
    
    double a = 1.0 / 3.0;
    double b = 0.333333;
    
    munit_assert_double_equal(a, b, 5);
    munit_assert_double(3.14, >, 3.0);
    munit_assert_float(2.5f, <, 3.0f);
    
    return MUNIT_OK;
}

static char* size_params[] = {"small", "medium", "large", NULL};
static char* type_params[] = {"int", "float", "double", NULL};

static MunitParameterEnum test_params[] = {
    {"size", size_params},
    {"type", type_params},
    {NULL, NULL}
};

static MunitResult test_parameterized_addition(const MunitParameter params[], void* data) {
    (void)data;
    
    const char* size = munit_parameters_get(params, "size");
    const char* type = munit_parameters_get(params, "type");
    
    if (strcmp(size, "small") == 0 && strcmp(type, "int") == 0) {
        munit_assert_int(add_host(1, 2), ==, 3);
    } else if (strcmp(size, "medium") == 0 && strcmp(type, "int") == 0) {
        munit_assert_int(add_host(100, 200), ==, 300);
    } else if (strcmp(size, "large") == 0 && strcmp(type, "int") == 0) {
        munit_assert_int(add_host(1000000, 2000000), ==, 3000000);
    } else if (strcmp(size, "small") == 0 && strcmp(type, "float") == 0) {
        munit_assert_float(1.5f + 2.5f, ==, 4.0f);
    } else if (strcmp(size, "medium") == 0 && strcmp(type, "float") == 0) {
        munit_assert_float(100.5f + 200.5f, ==, 301.0f);
    } else if (strcmp(size, "large") == 0 && strcmp(type, "float") == 0) {
        munit_assert_float(1000.123f + 2000.456f, >, 3000.0f);
    } else if (strcmp(size, "small") == 0 && strcmp(type, "double") == 0) {
        munit_assert_double(1.0 + 2.0, ==, 3.0);
    } else if (strcmp(size, "medium") == 0 && strcmp(type, "double") == 0) {
        munit_assert_double(100.0 + 200.0, ==, 300.0);
    } else if (strcmp(size, "large") == 0 && strcmp(type, "double") == 0) {
        munit_assert_double(1000000.0 + 2000000.0, ==, 3000000.0);
    }
    
    return MUNIT_OK;
}

static MunitTest tests[] = {
    {"/parameterized_addition", test_parameterized_addition, NULL, NULL, MUNIT_TEST_OPTION_NONE, test_params},
    {"/host_addition", test_host_addition, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/cuda_addition", test_cuda_addition, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/string_ops", test_string_operations, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/memory_ops", test_memory_operations, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/floating_point", test_floating_point, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL}
};

static const MunitSuite suite = {
    "/day066_tests", tests, NULL, 1, MUNIT_SUITE_OPTION_NONE
};

int main(int argc, char* argv[]) {
    return munit_suite_main(&suite, NULL, argc, argv);
}
