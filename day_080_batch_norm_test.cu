#include "day_080_batch_norm.cuh"
#include <catch2/catch_all.hpp>
#include <cmath>

TEST_CASE("Batch Norm - CUDA", "[cuda][batch_norm]") {
  SECTION("Simple case from Python example") {
    const int N = 3;
    const int C = 2;
    const float eps = 1e-5f;

    const float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float gamma[] = {1.0f, 1.0f};
    const float beta[] = {0.0f, 0.0f};
    float output[6] = {0};

    const float expected[] = {-1.224f, -1.224f, 0.0f, 0.0f, 1.224f, 1.224f};

    batch_norm_cuda_impl(input, gamma, beta, output, N, C, eps);

    for (int i = 0; i < N * C; i++) {
      REQUIRE(output[i] == Catch::Approx(expected[i]).epsilon(0.001));
    }
  }

  SECTION("Case with non-unit gamma and non-zero beta") {
    const int N = 2;
    const int C = 2;
    const float eps = 1e-5f;

    const float input[] = {0.0f, 1.0f, 2.0f, 3.0f};
    const float gamma[] = {2.0f, 0.5f};
    const float beta[] = {1.0f, -1.0f};
    float output[4] = {0};

    const float expected[] = {-1.0f, -1.5f, 3.0f, -0.5f};

    batch_norm_cuda_impl(input, gamma, beta, output, N, C, eps);

    for (int i = 0; i < N * C; i++) {
      REQUIRE(output[i] == Catch::Approx(expected[i]).epsilon(0.001));
    }
  }
}
