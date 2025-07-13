/*
 * NAME
 * 	sorted - verifies if a stream of numbers is sorted
 *
 * SYNOPSIS
 * 	sorted
 *
 * DESCRIPTION
 * 	Reads a stream of unsigned integers from stdin and prints
 * 	SORTED if the stream is sorted or NOT SORTED otherwise.
 *
 * USAGE
 * 	seq 1 10000 | shuf | sort -n | sorted
 *
 */
#include <stdio.h>

int main()
{
    int Num;
    int PreviousNum = -1; // HACK FOR NOW
    while (scanf("%d\n", &Num) != EOF)
    {
        if (Num < PreviousNum)
        {
            printf("NOT SORTED\n");
            return 1;
        }
        PreviousNum = Num;
    }
    printf("SORTED\n");
}
