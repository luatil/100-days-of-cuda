#include "day_079_subarray_sum.cuh"
#include <catch2/catch_all.hpp>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

TEST_CASE("Subarray Sum - CPU Naive", "[cpu][subarray_sum]") {
  SECTION("Simple case") {
    const int Input[] = {1, 2, 1, 3, 4};
    const int N = 5;
    const int S = 1;
    const int E = 3;
    int Output = 0;

    solve_subarray_sum_cpu_naive(Input, &Output, N, S, E);

    REQUIRE(Output == 6);
  }

  SECTION("Full array") {
    const int Input[] = {1, 2, 3, 4, 5};
    const int N = 5;
    const int S = 0;
    const int E = 4;
    int Output = 0;

    solve_subarray_sum_cpu_naive(Input, &Output, N, S, E);

    REQUIRE(Output == 15);
  }

  SECTION("Single element") {
    const int Input[] = {1, 2, 3, 4, 5};
    const int N = 5;
    const int S = 2;
    const int E = 2;
    int Output = 0;

    solve_subarray_sum_cpu_naive(Input, &Output, N, S, E);

    REQUIRE(Output == 3);
  }
}

TEST_CASE("Subarray Sum - GPU v1 Naive", "[cuda][gpu][subarray_sum]") {
  SECTION("Simple case") {
    const int Input[] = {1, 2, 1, 3, 4};
    const int N = 5;
    const int S = 1;
    const int E = 3;
    int Output = 0;

    solve_subarray_sum_gpu_v1_naive(Input, &Output, N, S, E);

    REQUIRE(Output == 6);
  }

  SECTION("Full array") {
    const int Input[] = {1, 2, 3, 4, 5};
    const int N = 5;
    const int S = 0;
    const int E = 4;
    int Output = 0;

    solve_subarray_sum_gpu_v1_naive(Input, &Output, N, S, E);

    REQUIRE(Output == 15);
  }

  SECTION("Single element") {
    const int Input[] = {1, 2, 3, 4, 5};
    const int N = 5;
    const int S = 2;
    const int E = 2;
    int Output = 0;

    solve_subarray_sum_gpu_v1_naive(Input, &Output, N, S, E);

    REQUIRE(Output == 3);
  }
}

TEST_CASE("Subarray Sum - Legacy", "[cuda][subarray_sum]") {
  SECTION("Simple case") {
    const int Input[] = {1, 2, 1, 3, 4};
    const int N = 5;
    const int S = 1;
    const int E = 3;
    int Output = 0;

    solve_subarray_sum(Input, &Output, N, S, E);

    REQUIRE(Output == 6);
  }
}
