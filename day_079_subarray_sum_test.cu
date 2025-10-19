#include "day_079_subarray_sum.cuh"
#include <catch2/catch_all.hpp>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

TEST_CASE("Subarray Sum", "[cuda][subarray_sum]") {
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
