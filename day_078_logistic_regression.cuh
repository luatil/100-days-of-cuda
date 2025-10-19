#ifndef DAY_078_LOGISTIC_REGRESSION_CUH
#define DAY_078_LOGISTIC_REGRESSION_CUH

// X, y, beta are device pointers
extern "C" void solve(const float *X, const float *y, float *beta,
                      int n_samples, int n_features);

#endif // DAY_078_LOGISTIC_REGRESSION_CUH
