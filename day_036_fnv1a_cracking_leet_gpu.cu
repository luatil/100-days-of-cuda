#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Simple string equality function
// Returns 1 if strings are equal, 0 if different
__device__ int Streq(const char *S1, const char *S2)
{
    // Handle null pointers
    if (S1 == NULL || S2 == NULL)
    {
        return S1 == S2; // Both null = equal, one null = not equal
    }

    // Compare character by character
    while (*S1 && *S2)
    {
        if (*S1 != *S2)
        {
            return 0; // Different characters found
        }
        S1++;
        S2++;
    }

    // Check if both strings ended at the same time
    return *S1 == *S2;
}

__device__ void DataFromTid(int Tid, int Length, char Result[8])
{
    int TempTid = Tid;
    for (int I = Length - 1; I >= 0; I--)
    {
        Result[I] = 'a' + (TempTid % 26);
        TempTid /= 26;
    }
    Result[Length] = 0;
}

// FNV-1a hash function that takes a byte array and its length as input
// Returns a 32-bit unsigned integer hash value
__device__ unsigned int Fnv1aHashBytes(char Data[8], int Length, int R)
{
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int Hash = OFFSET_BASIS;

    // First round: hash the input data
    for (int I = 0; I < Length; I++)
    {
        Hash = (Hash ^ Data[I]) * FNV_PRIME;
    }

    // Additional rounds: hash the previous hash result
    for (int Round = 1; Round < R; Round++)
    {
        unsigned int TempHash = Hash;
        Hash = OFFSET_BASIS;

        // Hash the 4 bytes of the previous hash (little-endian)
        for (int I = 0; I < 4; I++)
        {
            unsigned char Byte = (TempHash >> (I * 8)) & 0xFF;
            Hash = (Hash ^ Byte) * FNV_PRIME;
        }
    }

    return Hash;
}

__device__ void Copy(char *Dest, char Data[8], int Length)
{
    for (int I = 0; I < Length; I++)
    {
        Dest[I] = Data[I];
    }
    Dest[Length] = 0;
}

__global__ void CrackPasswordKernel(unsigned int TargetHash, int PasswordLength, int R, char *OutputPassword)
{
    const int TID = blockDim.x * blockIdx.x + threadIdx.x;
    char Data[8];
    DataFromTid(TID, PasswordLength, Data);
    unsigned int Hash = Fnv1aHashBytes(Data, PasswordLength, R);

    if (Hash == TargetHash)
    {
        Copy(OutputPassword, Data, PasswordLength);
    }
}

__host__ long long Power(int X, int N)
{
    if (N <= 0)
    {
        return 1;
    }
    else
    {
        return X * Power(X, N - 1);
    }
}

// output_password is a device pointer
void Solve(unsigned int TargetHash, int PasswordLength, int R, char *OutputPassword)
{
    int TotalValues = Power(26, PasswordLength);
    int BlockDim = 1024;
    int GridDim = (TotalValues + BlockDim - 1) / BlockDim;
    CrackPasswordKernel<<<GridDim, BlockDim>>>(TargetHash, PasswordLength, R, OutputPassword);
}
