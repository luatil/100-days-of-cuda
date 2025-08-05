/*
 *
 * NAME
 * 	fseq - generate random floating point numbers
 *
 * SYNOPSIS
 * 	fseq START END STEPS
 *
 * DESCRIPTION
 * 	Generates floating point numbers with the range [START, END] with
 * 	the [STEPS] number of values.
 *
 * USAGE
 * 	$ fseq 1.0 3.0 3
 * 	1.0
 * 	2.0
 * 	3.0
 *
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s START END STEPS\n", argv[0]);
        return 1;
    }

    float Start = atof(argv[1]);
    float End = atof(argv[2]);
    int Steps = atoi(argv[3]);

    if (Steps <= 0)
    {
        fprintf(stderr, "Error: STEPS must be positive\n");
        return 1;
    }

    if (Steps == 1)
    {
        printf("%g\n", Start);
        return 0;
    }

    float StepSize = (End - Start) / (Steps - 1);

    for (int I = 0; I < Steps; I++)
    {
        float Value = Start + I * StepSize;
        printf("%g\n", Value);
    }

    return 0;
}
