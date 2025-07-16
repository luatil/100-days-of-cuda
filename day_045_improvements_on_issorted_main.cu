/*
 * NAME
 * 	sorted - verifies if a stream of numbers is sorted
 *
 * SYNOPSIS
 * 	sorted
 *
 * DESCRIPTION
 * 	Reads a stream of unsigned integers from stdin and prints
 * 	the lines that were not sorted if the stream is sorted.
 *
 *
 * USAGE
 * 	seq 1 10000 | shuf | sort -n | sorted
 *
 */
#include <stdio.h>

int main()
{
    unsigned int Num;
    unsigned int PreviousNum = 0;
    while (scanf("%u\n", &Num) != EOF)
    {
        if (Num < PreviousNum)
        {
            printf("< %u\n", PreviousNum);
            printf("> %u\n", Num);
            return 1;
        }
        PreviousNum = Num;
    }
    printf("SORTED\n");
}
