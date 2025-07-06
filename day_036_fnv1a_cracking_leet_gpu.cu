#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Simple string equality function
// Returns 1 if strings are equal, 0 if different
__device__ int streq(const char *s1, const char *s2)
{
    // Handle null pointers
    if (s1 == NULL || s2 == NULL)
    {
        return s1 == s2; // Both null = equal, one null = not equal
    }

    // Compare character by character
    while (*s1 && *s2)
    {
        if (*s1 != *s2)
        {
            return 0; // Different characters found
        }
        s1++;
        s2++;
    }

    // Check if both strings ended at the same time
    return *s1 == *s2;
}

__device__ void data_from_tid(int tid, int length, char result[8])
{
    int temp_tid = tid;
    for (int i = length - 1; i >= 0; i--)
    {
        result[i] = 'a' + (temp_tid % 26);
        temp_tid /= 26;
    }
    result[length] = 0;
}

// FNV-1a hash function that takes a byte array and its length as input
// Returns a 32-bit unsigned integer hash value
__device__ unsigned int fnv1a_hash_bytes(char data[8], int length, int R)
{
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    // First round: hash the input data
    for (int i = 0; i < length; i++)
    {
        hash = (hash ^ data[i]) * FNV_PRIME;
    }

    // Additional rounds: hash the previous hash result
    for (int round = 1; round < R; round++)
    {
        unsigned int temp_hash = hash;
        hash = OFFSET_BASIS;

        // Hash the 4 bytes of the previous hash (little-endian)
        for (int i = 0; i < 4; i++)
        {
            unsigned char byte = (temp_hash >> (i * 8)) & 0xFF;
            hash = (hash ^ byte) * FNV_PRIME;
        }
    }

    return hash;
}

__device__ void copy(char *dest, char data[8], int length)
{
    for (int i = 0; i < length; i++)
    {
        dest[i] = data[i];
    }
    dest[length] = 0;
}

__global__ void crack_password_kernel(unsigned int target_hash, int password_length, int R, char *output_password)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    char data[8];
    data_from_tid(tid, password_length, data);
    unsigned int hash = fnv1a_hash_bytes(data, password_length, R);

    if (hash == target_hash)
    {
        copy(output_password, data, password_length);
    }
}

__host__ long long power(int x, int n)
{
    if (n <= 0)
    {
        return 1;
    }
    else
    {
        return x * power(x, n - 1);
    }
}

// output_password is a device pointer
void solve(unsigned int target_hash, int password_length, int R, char *output_password)
{
    int total_values = power(26, password_length);
    int block_dim = 1024;
    int grid_dim = (total_values + block_dim - 1) / block_dim;
    crack_password_kernel<<<grid_dim, block_dim>>>(target_hash, password_length, R, output_password);
}
