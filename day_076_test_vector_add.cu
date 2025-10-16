#include <catch2/catch_all.hpp>
#include "day_076_vector_add.cuh"

TEST_CASE("VectorAdd kernel tests", "[cuda][vector_add][example]")
{
  SECTION("Basic vector addition with N=1024")
  {
    const int N = 1024; float *A = (float*)malloc(sizeof(float)*N);
    float *B = (float*)malloc(sizeof(float)*N);
    float *C = (float*)malloc(sizeof(float)*N);

    // Initialize input vectors
    for (int I = 0; I < N; I++)
    {
      A[I] = 1.0f;
      B[I] = 2.0f;
    }

    size_t size = N * sizeof(float);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy inputs to device
    cudaError_t err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    REQUIRE(err == cudaSuccess);

    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    REQUIRE(err == cudaSuccess);

    // Launch kernel
    int BlockDim = 256;
    int GridDim = (N + BlockDim - 1) / BlockDim;
    VectorAdd<<<GridDim, BlockDim>>>(d_A, d_B, d_C, N);

    err = cudaGetLastError();
    REQUIRE(err == cudaSuccess);

    // Copy result back to host
    err = cudaMemcpy(C, d_C, sizeof(float)*N, cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);

    // Verify results
    for (int I = 0; I < N; I++)
    {
      REQUIRE(Abs(C[I] - 3.0f) < 0.01f);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
  }

  SECTION("Vector addition with different values")
  {
    const int N = 512;
    float *A = (float*)malloc(sizeof(float)*N);
    float *B = (float*)malloc(sizeof(float)*N);
    float *C = (float*)malloc(sizeof(float)*N);

    // Initialize with different values
    for (int I = 0; I < N; I++)
    {
      A[I] = (float)I;
      B[I] = (float)(I * 2);
    }

    size_t size = N * sizeof(float);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy inputs to device
    cudaError_t err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    REQUIRE(err == cudaSuccess);

    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    REQUIRE(err == cudaSuccess);

    // Launch kernel
    int BlockDim = 128;
    int GridDim = (N + BlockDim - 1) / BlockDim;
    VectorAdd<<<GridDim, BlockDim>>>(d_A, d_B, d_C, N);

    err = cudaGetLastError();
    REQUIRE(err == cudaSuccess);

    // Copy result back to host
    err = cudaMemcpy(C, d_C, sizeof(float)*N, cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);

    // Verify results: C[I] should equal I + (I*2) = I*3
    for (int I = 0; I < N; I++)
    {
      REQUIRE(Abs(C[I] - (float)(I * 3)) < 0.01f);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
  }
}
