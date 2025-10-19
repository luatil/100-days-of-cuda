#include "day_078_logistic_regression.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// Kernel to compute matrix-vector product: result = X * beta
// X: n_samples x n_features (row-major)
// beta: n_features x 1
// result: n_samples x 1
__global__ void matmul_kernel(const float *X, const float *beta, float *result,
                               int n_samples, int n_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_samples) {
    float sum = 0.0f;
    for (int j = 0; j < n_features; j++) {
      sum += X[idx * n_features + j] * beta[j];
    }
    result[idx] = sum;
  }
}

// Kernel to apply sigmoid function: p = 1 / (1 + exp(-z))
__global__ void sigmoid_kernel(const float *z, float *p, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    p[idx] = 1.0f / (1.0f + expf(-z[idx]));
  }
}

// Kernel to compute gradient: gradient = X^T * (p - y) + lambda * beta
// X: n_samples x n_features (row-major)
// diff: p - y (n_samples x 1)
// gradient: n_features x 1
__global__ void gradient_kernel(const float *X, const float *p, const float *y,
                                 const float *beta, float *gradient,
                                 int n_samples, int n_features, float lambda) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_features) {
    float sum = 0.0f;
    for (int i = 0; i < n_samples; i++) {
      float diff = p[i] - y[i];
      sum += X[i * n_features + idx] * diff;
    }
    gradient[idx] = sum + lambda * beta[idx];  // Add L2 regularization
  }
}

// Kernel to update beta: beta = beta - learning_rate * gradient
__global__ void update_kernel(float *beta, const float *gradient,
                               float learning_rate, int n_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_features) {
    beta[idx] -= learning_rate * gradient[idx];
  }
}

// Kernel to compute L2 norm squared of gradient
__global__ void norm_squared_kernel(const float *gradient, float *partial_sums,
                                     int n_features) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  sdata[tid] = (idx < n_features) ? gradient[idx] * gradient[idx] : 0.0f;
  __syncthreads();

  // Reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    partial_sums[blockIdx.x] = sdata[0];
  }
}

extern "C" void solve(const float *X, const float *y, float *beta,
                      int n_samples, int n_features) {
  const int max_iterations = 10000;
  const float learning_rate = 0.01f;
  const float tolerance = 1e-2f;
  const float lambda = 0.0f;  // L2 regularization parameter
  const int block_size = 256;

  // Initialize beta to zeros
  cudaMemset(beta, 0, n_features * sizeof(float));

  // Allocate temporary device memory
  float *d_z;           // n_samples x 1: X * beta
  float *d_p;           // n_samples x 1: sigmoid(z)
  float *d_gradient;    // n_features x 1
  float *d_norm_partial; // For norm computation

  cudaMalloc(&d_z, n_samples * sizeof(float));
  cudaMalloc(&d_p, n_samples * sizeof(float));
  cudaMalloc(&d_gradient, n_features * sizeof(float));

  int norm_blocks = (n_features + block_size - 1) / block_size;
  cudaMalloc(&d_norm_partial, norm_blocks * sizeof(float));

  // Grid dimensions
  int grid_samples = (n_samples + block_size - 1) / block_size;
  int grid_features = (n_features + block_size - 1) / block_size;

  // Gradient descent loop
  for (int iter = 0; iter < max_iterations; iter++) {
    // 1. Compute z = X * beta
    matmul_kernel<<<grid_samples, block_size>>>(X, beta, d_z, n_samples, n_features);

    // 2. Compute p = sigmoid(z)
    sigmoid_kernel<<<grid_samples, block_size>>>(d_z, d_p, n_samples);

    // 3. Compute gradient = X^T * (p - y) + lambda * beta
    gradient_kernel<<<grid_features, block_size>>>(X, d_p, y, beta, d_gradient,
                                                    n_samples, n_features, lambda);

    // 4. Check convergence: compute ||gradient||^2
    norm_squared_kernel<<<norm_blocks, block_size, block_size * sizeof(float)>>>(
        d_gradient, d_norm_partial, n_features);

    // Sum partial results on CPU (simple approach for convergence check)
    float h_norm_partial[norm_blocks];
    cudaMemcpy(h_norm_partial, d_norm_partial, norm_blocks * sizeof(float),
               cudaMemcpyDeviceToHost);

    float norm_sq = 0.0f;
    for (int i = 0; i < norm_blocks; i++) {
      norm_sq += h_norm_partial[i];
    }
    float norm = sqrtf(norm_sq);


    // 5. Update beta = beta - learning_rate * gradient
    update_kernel<<<grid_features, block_size>>>(beta, d_gradient, learning_rate,
                                                  n_features);

    // Check convergence
    if (norm < tolerance) {
      break;
    }
  }

  // Cleanup
  cudaFree(d_z);
  cudaFree(d_p);
  cudaFree(d_gradient);
  cudaFree(d_norm_partial);
}
