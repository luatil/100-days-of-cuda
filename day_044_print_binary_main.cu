/*
 * NAME
 * 	pbin - print sequence of numbers in binary format
 *
 * SYNOPSIS
 * 	pbin
 *
 * DESCRIPTION
 * 	Reads unsigned integer numbers from stdin and prints them in
 * 	binary format e.g. 1010101 format.
 *
 * USAGE
 * 	- Print the first 10 numbers in binary format
 * 	seq 1 10 | pbin
 */
#include <stdio.h>

static void PrintBinary(unsigned int Num, int Width)
{
    for (int I = 0; I < Width; I++)
    {
	printf("%d", (Num >> (31 - I)) & 1);
    }
    printf("\n");
}

int main()
{
    unsigned int Num;
    while (scanf("%u\n", &Num) != EOF)
    {
	PrintBinary(Num, 8);
    }
}
