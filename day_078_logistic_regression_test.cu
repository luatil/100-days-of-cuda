#include <catch2/catch_all.hpp>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include "day_078_logistic_regression.cuh"

TEST_CASE("Logistic Regression", "[cuda][logistic_regression]") {
  SECTION("Simple case") {
    float X[8][2] = {{2.0, 1.0},   {1.0, 2.0},   {3.0, 3.0},   {1.5, 2.5},
                     {-1.0, -2.0}, {-2.0, -1.0}, {-1.5, -2.5}, {-3.0, -3.0}};
    float y[8] = {1, 1, 1, 0, 0, 0, 1, 0};
    float beta[2] = {0.0f, 0.0f};

    int n_samples = 8;
    int n_features = 2;

    // Flatten X into 1D array for the C function
    float X_flat[16];
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 2; j++) {
        X_flat[i * 2 + j] = X[i][j];
      }
    }

    // Allocate device memory
    float *d_X, *d_y, *d_beta;
    cudaMalloc(&d_X, n_samples * n_features * sizeof(float));
    cudaMalloc(&d_y, n_samples * sizeof(float));
    cudaMalloc(&d_beta, n_features * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_X, X_flat, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, n_features * sizeof(float), cudaMemcpyHostToDevice);

    // Call solve with device pointers
    solve(d_X, d_y, d_beta, n_samples, n_features);

    // Copy result back to host
    cudaMemcpy(beta, d_beta, n_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_beta);

    // Expected output: beta ≈ [2.2562716007232666, -1.2880022525787354]
    REQUIRE(std::abs(beta[0] - 2.2562716f) < 0.001f);
    REQUIRE(std::abs(beta[1] - (-1.2880023f)) < 0.001f);
  }
}
