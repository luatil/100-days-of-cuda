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

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s START END STEPS\n", argv[0]);
        return 1;
    }

    float start = atof(argv[1]);
    float end = atof(argv[2]);
    int steps = atoi(argv[3]);

    if (steps <= 0) {
        fprintf(stderr, "Error: STEPS must be positive\n");
        return 1;
    }

    if (steps == 1) {
        printf("%g\n", start);
        return 0;
    }

    float step_size = (end - start) / (steps - 1);
    
    for (int i = 0; i < steps; i++) {
        float value = start + i * step_size;
        printf("%g\n", value);
    }

    return 0;
}
