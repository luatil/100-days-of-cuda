#include <catch2/catch_all.hpp>
#include <random>
#include "day_076_vector_add.cuh"

TEST_CASE("VectorAdd property-based tests", "[cuda][vector_add][property]")
{
  // Property 1: Commutativity - A + B = B + A
  SECTION("Commutativity: A + B = B + A")
  {
    auto N = GENERATE(128, 256, 512, 1024, 2048);
    auto seed = GENERATE(take(5, random(0, 10000)));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    float *A = (float*)malloc(sizeof(float)*N);
    float *B = (float*)malloc(sizeof(float)*N);
    float *C1 = (float*)malloc(sizeof(float)*N);
    float *C2 = (float*)malloc(sizeof(float)*N);

    // Generate random inputs
    for (int I = 0; I < N; I++)
    {
      A[I] = dist(rng);
      B[I] = dist(rng);
    }

    size_t size = N * sizeof(float);
    float *d_A, *d_B, *d_C1, *d_C2;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C1, size);
    cudaMalloc(&d_C2, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int BlockDim = 256;
    int GridDim = (N + BlockDim - 1) / BlockDim;

    // Compute A + B
    VectorAdd<<<GridDim, BlockDim>>>(d_A, d_B, d_C1, N);
    // Compute B + A
    VectorAdd<<<GridDim, BlockDim>>>(d_B, d_A, d_C2, N);

    cudaMemcpy(C1, d_C1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(C2, d_C2, size, cudaMemcpyDeviceToHost);

    // Verify commutativity
    for (int I = 0; I < N; I++)
    {
      REQUIRE(Abs(C1[I] - C2[I]) < 1e-5f);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C1); cudaFree(d_C2);
    free(A); free(B); free(C1); free(C2);
  }

  // Property 2: Identity - A + 0 = A
  SECTION("Identity: A + 0 = A")
  {
    auto N = GENERATE(128, 512, 1024);
    auto seed = GENERATE(take(3, random(0, 10000)));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    float *A = (float*)malloc(sizeof(float)*N);
    float *Zero = (float*)malloc(sizeof(float)*N);
    float *C = (float*)malloc(sizeof(float)*N);

    for (int I = 0; I < N; I++)
    {
      A[I] = dist(rng);
      Zero[I] = 0.0f;
    }

    size_t size = N * sizeof(float);
    float *d_A, *d_Zero, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_Zero, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Zero, Zero, size, cudaMemcpyHostToDevice);

    int BlockDim = 256;
    int GridDim = (N + BlockDim - 1) / BlockDim;
    VectorAdd<<<GridDim, BlockDim>>>(d_A, d_Zero, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    for (int I = 0; I < N; I++)
    {
      REQUIRE(Abs(C[I] - A[I]) < 1e-5f);
    }

    cudaFree(d_A); cudaFree(d_Zero); cudaFree(d_C);
    free(A); free(Zero); free(C);
  }

  // Property 3: Associativity - (A + B) + C = A + (B + C)
  SECTION("Associativity: (A + B) + C = A + (B + C)")
  {
    auto N = GENERATE(256, 1024);
    auto seed = GENERATE(take(3, random(0, 10000)));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    float *A = (float*)malloc(sizeof(float)*N);
    float *B = (float*)malloc(sizeof(float)*N);
    float *C = (float*)malloc(sizeof(float)*N);
    float *Temp1 = (float*)malloc(sizeof(float)*N);
    float *Temp2 = (float*)malloc(sizeof(float)*N);
    float *Result1 = (float*)malloc(sizeof(float)*N);
    float *Result2 = (float*)malloc(sizeof(float)*N);

    for (int I = 0; I < N; I++)
    {
      A[I] = dist(rng);
      B[I] = dist(rng);
      C[I] = dist(rng);
    }

    size_t size = N * sizeof(float);
    float *d_A, *d_B, *d_C, *d_Temp1, *d_Temp2, *d_Result1, *d_Result2;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_Temp1, size);
    cudaMalloc(&d_Temp2, size);
    cudaMalloc(&d_Result1, size);
    cudaMalloc(&d_Result2, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    int BlockDim = 256;
    int GridDim = (N + BlockDim - 1) / BlockDim;

    // Compute (A + B) + C
    VectorAdd<<<GridDim, BlockDim>>>(d_A, d_B, d_Temp1, N);
    VectorAdd<<<GridDim, BlockDim>>>(d_Temp1, d_C, d_Result1, N);

    // Compute A + (B + C)
    VectorAdd<<<GridDim, BlockDim>>>(d_B, d_C, d_Temp2, N);
    VectorAdd<<<GridDim, BlockDim>>>(d_A, d_Temp2, d_Result2, N);

    cudaMemcpy(Result1, d_Result1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Result2, d_Result2, size, cudaMemcpyDeviceToHost);

    for (int I = 0; I < N; I++)
    {
      REQUIRE(Abs(Result1[I] - Result2[I]) < 1e-4f);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_Temp1); cudaFree(d_Temp2);
    cudaFree(d_Result1); cudaFree(d_Result2);
    free(A); free(B); free(C);
    free(Temp1); free(Temp2);
    free(Result1); free(Result2);
  }
}
