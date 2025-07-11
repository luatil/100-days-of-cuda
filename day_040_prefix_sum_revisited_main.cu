/*
 * NAME
 * 	psum - calculates prefix sum
 *
 * SYNOPSIS
 * 	psum [OPTION]... [FILE]
 *
 * DESCRIPTION
 * 	Calculates the prefix sum from input lines to standard output.
 * 	Defaults to using the GPU if a supported one is available.
 *
 * 	-e, --exclusive
 * 		calculates exclusive scan
 *
 * 	-c, --cpu
 * 		use the cpu
 *
 * 	-g, --gpu
 * 		use the gpu
 *
 * 	--algorithm=
 * 		Defines the algorithm to be used to calculate the prefix sum.
 * 		Only to be used when the prefix sum is executed on the GPU.
 *
 * 		A value for this option must be provided; possible ones are
 *
 * 		blelloch
 * 			work efficient algorithm for computing the prefix sum
 *
 *		hillis-steele
 *			simpler version of the prefix sum algorithm
 *
 *		brent-kung
 *			a work-efficient algorithm similar to blelloch but
 *			with different tree traversal patterns
 *
 * 	--help display this help and exit
 *
 */
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef enum
{
    ALGORITHM_BLELLOCH,
    ALGORITHM_HILLIS_STEELE,
    ALGORITHM_BRENT_KUNG
} algorithm;

typedef struct
{
    int Exclusive;
    int UseCPU;
    int UseGPU;
    int Verbose;
    algorithm Algorithm;
    char *InputFile;
} options;

static void PrintUsage(const char *ProgramName)
{
    printf("Usage: %s [OPTION]... [FILE]\n", ProgramName);
    printf("Calculates the prefix sum from input lines to standard output.\n");
    printf("Defaults to using the GPU if a supported one is available.\n\n");
    printf("Options:\n");
    printf("  -e, --exclusive        calculates exclusive scan\n");
    printf("  -c, --cpu              use the cpu\n");
    printf("  -g, --gpu              use the gpu\n");
    printf("  -v, --verbose          print used options\n");
    printf("      --algorithm=ALG    algorithm to use (blelloch, hillis-steele, brent-kung)\n");
    printf("      --help             display this help and exit\n");
}

static algorithm ParseAlgorithm(const char *AlgorithmStr)
{
    if (strcmp(AlgorithmStr, "blelloch") == 0)
    {
        return ALGORITHM_BLELLOCH;
    }
    else if (strcmp(AlgorithmStr, "hillis-steele") == 0)
    {
        return ALGORITHM_HILLIS_STEELE;
    }
    else if (strcmp(AlgorithmStr, "brent-kung") == 0)
    {
        return ALGORITHM_BRENT_KUNG;
    }
    else
    {
        fprintf(stderr, "Error: Invalid algorithm '%s'\n", AlgorithmStr);
        fprintf(stderr, "Valid algorithms: blelloch, hillis-steele, brent-kung\n");
        exit(EXIT_FAILURE);
    }
}

static options ParseCommandLine(int ArgumentCount, char *Arguments[])
{
    options Opts = {0, 0, 0, 0, ALGORITHM_BLELLOCH, NULL};

    static struct option LongOptions[] = {{"exclusive", no_argument, 0, 'e'},
                                          {"cpu", no_argument, 0, 'c'},
                                          {"gpu", no_argument, 0, 'g'},
                                          {"verbose", no_argument, 0, 'v'},
                                          {"algorithm", required_argument, 0, 'a'},
                                          {"help", no_argument, 0, 'h'},
                                          {0, 0, 0, 0}};

    int OptionIndex = 0;
    int Char;

    while ((Char = getopt_long(ArgumentCount, Arguments, "ecgv", LongOptions, &OptionIndex)) != -1)
    {
        switch (Char)
        {
        case 'e':
            Opts.Exclusive = 1;
            break;
        case 'c':
            Opts.UseCPU = 1;
            break;
        case 'g':
            Opts.UseGPU = 1;
            break;
        case 'v':
            Opts.Verbose = 1;
            break;
        case 'a':
            Opts.Algorithm = ParseAlgorithm(optarg);
            break;
        case 'h':
            PrintUsage(Arguments[0]);
            exit(EXIT_SUCCESS);
            break;
        case '?':
            exit(EXIT_FAILURE);
            break;
        default:
            abort();
        }
    }

    if (Opts.UseCPU && Opts.UseGPU)
    {
        fprintf(stderr, "Error: Cannot specify both --cpu and --gpu\n");
        exit(EXIT_FAILURE);
    }

    if (!Opts.UseCPU && !Opts.UseGPU)
    {
        Opts.UseGPU = 1;
    }

    if (optind < ArgumentCount)
    {
        Opts.InputFile = Arguments[optind];
    }

    return Opts;
}

static void PrintOptions(const options *Opts)
{
    printf("Used options:\n");
    printf("  Scan type: %s\n", Opts->Exclusive ? "exclusive" : "inclusive");
    printf("  Execution: %s\n", Opts->UseCPU ? "CPU" : "GPU");

    if (!Opts->UseCPU)
    {
        const char *AlgorithmName;
        switch (Opts->Algorithm)
        {
        case ALGORITHM_BLELLOCH:
            AlgorithmName = "blelloch";
            break;
        case ALGORITHM_HILLIS_STEELE:
            AlgorithmName = "hillis-steele";
            break;
        case ALGORITHM_BRENT_KUNG:
            AlgorithmName = "brent-kung";
            break;
        default:
            AlgorithmName = "unknown";
            break;
        }
        printf("  Algorithm: %s\n", AlgorithmName);
    }

    printf("  Input: %s\n", Opts->InputFile ? Opts->InputFile : "stdin");
    printf("\n");
}

static int *ReadIntegersBulk(FILE *InputFile, int *Count)
{
    long FileSize = 0;
    char *Buffer = NULL;

    if (InputFile != stdin)
    {
        if (fseek(InputFile, 0, SEEK_END) != 0)
        {
            fprintf(stderr, "Error: Could not seek to end of file\n");
            return NULL;
        }

        FileSize = ftell(InputFile);
        if (FileSize == -1)
        {
            fprintf(stderr, "Error: Could not get file size\n");
            return NULL;
        }

        if (fseek(InputFile, 0, SEEK_SET) != 0)
        {
            fprintf(stderr, "Error: Could not seek to beginning of file\n");
            return NULL;
        }

        Buffer = (char *)malloc(FileSize + 1);
        if (Buffer == NULL)
        {
            fprintf(stderr, "Error: Could not allocate memory for file buffer\n");
            return NULL;
        }

        size_t BytesRead = fread(Buffer, 1, FileSize, InputFile);
        if (BytesRead != (size_t)FileSize)
        {
            fprintf(stderr, "Error: Could not read entire file\n");
            free(Buffer);
            return NULL;
        }
        Buffer[FileSize] = '\0';
    }
    else
    {
        size_t BufferSize = 8192;
        size_t TotalRead = 0;
        Buffer = (char *)malloc(BufferSize);
        if (Buffer == NULL)
        {
            fprintf(stderr, "Error: Could not allocate memory for stdin buffer\n");
            return NULL;
        }

        size_t BytesRead;
        while ((BytesRead = fread(Buffer + TotalRead, 1, BufferSize - TotalRead - 1, InputFile)) > 0)
        {
            TotalRead += BytesRead;
            if (TotalRead >= BufferSize - 1)
            {
                BufferSize *= 2;
                char *NewBuffer = (char *)realloc(Buffer, BufferSize);
                if (NewBuffer == NULL)
                {
                    fprintf(stderr, "Error: Could not reallocate memory for stdin buffer\n");
                    free(Buffer);
                    return NULL;
                }
                Buffer = NewBuffer;
            }
        }
        Buffer[TotalRead] = '\0';
        FileSize = TotalRead;
    }

    int Capacity = 1024;
    int *Numbers = (int *)malloc(Capacity * sizeof(int));
    if (Numbers == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for numbers array\n");
        free(Buffer);
        return NULL;
    }

    *Count = 0;
    char *Current = Buffer;
    char *End;

    while (*Current != '\0')
    {
        while (*Current == ' ' || *Current == '\t' || *Current == '\n' || *Current == '\r')
        {
            Current++;
        }

        if (*Current == '\0')
        {
            break;
        }

        long Value = strtol(Current, &End, 10);
        if (End == Current)
        {
            fprintf(stderr, "Error: Invalid number format in input\n");
            free(Buffer);
            free(Numbers);
            return NULL;
        }

        if (*Count >= Capacity)
        {
            Capacity *= 2;
            int *NewNumbers = (int *)realloc(Numbers, Capacity * sizeof(int));
            if (NewNumbers == NULL)
            {
                fprintf(stderr, "Error: Could not reallocate memory for numbers array\n");
                free(Buffer);
                free(Numbers);
                return NULL;
            }
            Numbers = NewNumbers;
        }

        Numbers[*Count] = (int)Value;
        (*Count)++;
        Current = End;
    }

    free(Buffer);
    return Numbers;
}

int main(int argc, char *argv[])
{
    options Opts = ParseCommandLine(argc, argv);

    if (Opts.Verbose)
    {
        PrintOptions(&Opts);
    }

    struct timespec StartTime, EndTime;
    clock_gettime(CLOCK_MONOTONIC, &StartTime);

    FILE *InputFile = stdin;
    if (Opts.InputFile)
    {
        InputFile = fopen(Opts.InputFile, "r");
        if (InputFile == NULL)
        {
            fprintf(stderr, "Error: Cannot open file '%s': ", Opts.InputFile);
            perror("");
            exit(EXIT_FAILURE);
        }
    }

    int Count = 0;
    int *Numbers = ReadIntegersBulk(InputFile, &Count);

    if (Numbers == NULL)
    {
        if (Opts.InputFile)
        {
            fclose(InputFile);
        }
        exit(EXIT_FAILURE);
    }

    struct timespec ReadEndTime;
    clock_gettime(CLOCK_MONOTONIC, &ReadEndTime);

    if (Opts.Verbose)
    {
        double ReadTime = (ReadEndTime.tv_sec - StartTime.tv_sec) + (ReadEndTime.tv_nsec - StartTime.tv_nsec) / 1e9;
        fprintf(stderr, "File Reading     : %.6f sec | %8.2f MB/s | %d integers\n", ReadTime,
                (Count * sizeof(int)) / (ReadTime * 1024 * 1024), Count);
    }

    int *Results = (int *)malloc(Count * sizeof(int));
    if (Results == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for results array\n");
        free(Numbers);
        if (Opts.InputFile)
        {
            fclose(InputFile);
        }
        exit(EXIT_FAILURE);
    }

    struct timespec PrefixStartTime;
    clock_gettime(CLOCK_MONOTONIC, &PrefixStartTime);

    int Sum = 0;
    for (int i = 0; i < Count; i++)
    {
        Sum += Numbers[i];
        Results[i] = Sum;
    }

    struct timespec PrefixEndTime;
    clock_gettime(CLOCK_MONOTONIC, &PrefixEndTime);

    struct timespec OutputStartTime;
    clock_gettime(CLOCK_MONOTONIC, &OutputStartTime);

    const size_t BUFFER_SIZE = 65536;
    char *OutputBuffer = (char *)malloc(BUFFER_SIZE);
    if (OutputBuffer == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for output buffer\n");
        free(Numbers);
        free(Results);
        if (Opts.InputFile)
        {
            fclose(InputFile);
        }
        exit(EXIT_FAILURE);
    }

    size_t BufferPos = 0;
    for (int i = 0; i < Count; i++)
    {
        int BytesWritten = snprintf(OutputBuffer + BufferPos, BUFFER_SIZE - BufferPos, "%d\n", Results[i]);

        if (BytesWritten < 0)
        {
            fprintf(stderr, "Error: Failed to format output\n");
            free(OutputBuffer);
            free(Numbers);
            free(Results);
            if (Opts.InputFile)
            {
                fclose(InputFile);
            }
            exit(EXIT_FAILURE);
        }

        BufferPos += BytesWritten;

        if (BufferPos > BUFFER_SIZE - 20)
        {
            if (fwrite(OutputBuffer, 1, BufferPos, stdout) != BufferPos)
            {
                fprintf(stderr, "Error: Failed to write to output\n");
                free(OutputBuffer);
                free(Numbers);
                free(Results);
                if (Opts.InputFile)
                {
                    fclose(InputFile);
                }
                exit(EXIT_FAILURE);
            }
            BufferPos = 0;
        }
    }

    if (BufferPos > 0)
    {
        if (fwrite(OutputBuffer, 1, BufferPos, stdout) != BufferPos)
        {
            fprintf(stderr, "Error: Failed to write final buffer to output\n");
            free(OutputBuffer);
            free(Numbers);
            free(Results);
            if (Opts.InputFile)
            {
                fclose(InputFile);
            }
            exit(EXIT_FAILURE);
        }
    }

    free(OutputBuffer);
    struct timespec OutputEndTime;
    clock_gettime(CLOCK_MONOTONIC, &OutputEndTime);

    clock_gettime(CLOCK_MONOTONIC, &EndTime);

    if (Opts.Verbose)
    {
        double TotalTime = (EndTime.tv_sec - StartTime.tv_sec) + (EndTime.tv_nsec - StartTime.tv_nsec) / 1e9;
        double PrefixTime =
            (PrefixEndTime.tv_sec - PrefixStartTime.tv_sec) + (PrefixEndTime.tv_nsec - PrefixStartTime.tv_nsec) / 1e9;
        double OutputTime =
            (OutputEndTime.tv_sec - OutputStartTime.tv_sec) + (OutputEndTime.tv_nsec - OutputStartTime.tv_nsec) / 1e9;
        double TotalDataMB = (Count * sizeof(int) * 2) / (1024.0 * 1024.0);

        fprintf(stderr, "Prefix Sum       : %.6f sec | %8.2f MB/s | %d operations\n", PrefixTime,
                (Count * sizeof(int) * 2) / (PrefixTime * 1024 * 1024), Count);
        fprintf(stderr, "File Writing     : %.6f sec | %8.2f MB/s | %d integers\n", OutputTime,
                (Count * sizeof(int)) / (OutputTime * 1024 * 1024), Count);
        fprintf(stderr, "Total Time       : %.6f sec | %8.2f MB/s | %d integers\n", TotalTime, TotalDataMB / TotalTime,
                Count);
        fprintf(stderr, "Memory Usage     : %.2f MB total | input + output arrays\n", TotalDataMB);
    }

    free(Numbers);
    free(Results);
    if (Opts.InputFile)
    {
        fclose(InputFile);
    }
}
