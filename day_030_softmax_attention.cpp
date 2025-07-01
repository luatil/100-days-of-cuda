#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <stdio.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define Min(a, b) ((a) < (b) ? (a) : (b))

/*
* Implement a CUDA program that computes the softmax attention operation
* for a given set of matrices. Given the query matrix Q of size M×d,
* key matrix K of size N×d, and value matrix V of size N×d, your program
* should compute the output matrix using the formula:
*/

// Attention(Q, K, V) = softmax(Q*K.T/sqrt(d)) * V
// Where:
//  Q: Mxd
//  K: Nxd
//  V: Nxd
// And the softmax function is applied row-wise
static void AttentionKernel(float *Query, float *Key, float *Value, float *Out, int M, int N, int d)
{
    // Let's start with Q*K.T * V - this will result in a MxN matrix that we can output to the console
    // and start working on.
    //
    // Output will be Mxd
    //
    // Do I need to materialize the internal matrix?

    // Considering here that I need to allocate a temporary matrix of MxN to hold the value of Q*K.T
    float *Internal = (float*)malloc(sizeof(float)*M*N);

    for(int I = 0; I < M; I++)
    {
        for (int J = 0; J < N; J++)
        {
            float Sum = 0.0f;
            for (int K = 0; K < d; K++)
            {
                Sum += Query[I*d+K] * Key[J*d+K];
            }
            Internal[I*N+J] = Sum / sqrtf((float)d);
        }
    }

    // Now we need to calculate modify internal such that it has the softmax value
    // for softmax we will use the f(x_i) = e^x_i-max_x/ sum(e^x_i-max_x) for all x
    // So we need to iterate over the three times:
    // 1. Calculate the max value
    // 2. Calculate the exponential sum
    // 3. Divide each element by the exponential sum

    for (float *Row = Internal; (Row - Internal) < M*N; Row += N)
    {
        float MaxValue = -FLT_MAX;
        for (float *Col = Row; (Col - Row) < N; Col++)
        {
            MaxValue = Max(MaxValue, *Col);
        }
        float ExpSum = 0.0f;
        for (float *Col = Row; (Col - Row) < N; Col++)
        {
            *Col = expf(*Col - MaxValue);
            ExpSum += *Col;
        }
        for (float *Col = Row; (Col - Row) < N; Col++)
        {
            *Col /= ExpSum;
        }
    }

    // Now we just need to do another matmul to get to the output value
    for(int I = 0; I < M; I++)
    {
        for (int J = 0; J < d; J++)
        {
            float Sum = 0.0f;
            for (int K = 0; K < N; K++)
            {
                Sum += Internal[I*N+K] * Value[K*d+J];
            }
            Out[I*d+J] = Sum;
        }
    }

    if (Internal)
    {
        free(Internal);
    }
}

static void example_01()
{
    // Input matrices
    float Q[2][4] = {
	{1.0, 0.0, 0.0, 0.0},
	{0.0, 1.0, 0.0, 0.0}
    };

    float K[3][4] = {
	{1.0, 0.0, 0.0, 0.0},
	{0.0, 1.0, 0.0, 0.0},
	{0.0, 0.0, 1.0, 0.0}
    };

    float V[3][4] = {
	{1.0, 2.0, 3.0, 4.0},
	{5.0, 6.0, 7.0, 8.0},
	{9.0, 10.0, 11.0, 12.0}
    };

    // Output matrix
    float EO[2][4] = {
	{4.29, 5.29, 6.29, 7.29},
	{5.00, 6.00, 7.00, 8.00}
    };

    float Output[2][4] = {};

    AttentionKernel((float*)Q, (float*)K, (float*)V, (float*)Output, 2, 3, 4);

    // Print the output matrix
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", Output[i][j]);
        }
        printf("\n");
    }
}


static void example_02()
{
    // Input matrices
    float Q[1][2] = {
        {1.0f, 2.0f}
    };

    float K[2][2] = {
	{1.0, 0.0},
	{0.0, 1.0},
    };

    float V[2][2] = {
	{3.0, 4.0},
	{5.0, 6.0},
    };

    // Output matrix
    float EO[1][2] = {
	{4.34, 5.34},
    };

    float Output[1][2] = {};

    AttentionKernel((float*)Q, (float*)K, (float*)V, (float*)Output, 1, 2, 2);

    // Print the output matrix
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%f ", Output[i][j]);
        }
        printf("\n");
    }
}

int main()
{
    // Print first example
    printf("Example 1:\n");
    example_01();
    printf("\nExample 2:\n");
    example_02();
}
