#include "day_079_subarray_sum.cuh"
#include <stdio.h>

// Input: input = [1, 2, 1, 3, 4], S = 1, E = 3
// Output: output = 6

int main() {
  const int Input[] = {1, 2, 1, 3, 4};
  const int N = 5;
  const int S = 1;
  const int E = 3;
  int Output = 0;

  solve_subarray_sum(Input, &Output, N, S, E);

  for (int I = 0; I < N; I++) {
    printf("%d\n", Input[I]);
  }

  printf("The subarray sum is: \n");

  printf("%d\n", Output);
}
