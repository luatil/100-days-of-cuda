/*
 * NAME
 * 	imconv - applies 2D convolution filters to images
 *
 * SYNOPSIS
 * 	imconv [OPTION]... INPUT_FILE OUTPUT_FILE
 *
 * DESCRIPTION
 * 	Applies convolution filters to images using CUDA acceleration.
 * 	Supports common image formats via stb_image.
 *
 * 	-f, --filter=FILTER
 * 		specify the convolution filter to apply
 * 		Available filters: edge, sharpen, emboss, blur, identity
 *
 * 	-s, --scale=SCALE
 * 		scaling factor applied to convolution result (default: 1.0)
 *
 * 	-b, --bias=BIAS
 * 		bias added to convolution result (default: 0.0)
 *
 * 	-v, --verbose
 * 		print timing and performance information
 *
 * 	--help
 * 		display this help and exit
 *
 */
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "day_003_vendor_libraries.h"

#include "day_041_convolution_kernel.cu"

typedef enum
{
    FILTER_EDGE,
    FILTER_SHARPEN,
    FILTER_EMBOSS,
    FILTER_BLUR,
    FILTER_IDENTITY
} filter_type;

typedef struct
{
    filter_type Filter;
    float Scale;
    float Bias;
    int Verbose;
    char *InputFile;
    char *OutputFile;
} options;

// Predefined convolution kernels
float EdgeKernel[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};

float SharpenKernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

float EmbossKernel[9] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};

float BlurKernel[9] = {1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9};

float IdentityKernel[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};

static void PrintUsage(const char *ProgramName)
{
    printf("Usage: %s [OPTION]... INPUT_FILE OUTPUT_FILE\n", ProgramName);
    printf("Applies convolution filters to images using CUDA acceleration.\n\n");
    printf("Options:\n");
    printf("  -f, --filter=FILTER    filter to apply (edge, sharpen, emboss, blur, identity)\n");
    printf("  -s, --scale=SCALE      scaling factor (default: 1.0)\n");
    printf("  -b, --bias=BIAS        bias value (default: 0.0)\n");
    printf("  -v, --verbose          print timing information\n");
    printf("      --help             display this help and exit\n");
}

static filter_type ParseFilter(const char *FilterStr)
{
    if (strcmp(FilterStr, "edge") == 0)
        return FILTER_EDGE;
    if (strcmp(FilterStr, "sharpen") == 0)
        return FILTER_SHARPEN;
    if (strcmp(FilterStr, "emboss") == 0)
        return FILTER_EMBOSS;
    if (strcmp(FilterStr, "blur") == 0)
        return FILTER_BLUR;
    if (strcmp(FilterStr, "identity") == 0)
        return FILTER_IDENTITY;

    fprintf(stderr, "Error: Unknown filter '%s'\n", FilterStr);
    fprintf(stderr, "Available filters: edge, sharpen, emboss, blur, identity\n");
    exit(EXIT_FAILURE);
}

static options ParseCommandLine(int Argc, char *Argv[])
{
    options Opts = {FILTER_EDGE, 1.0f, 0.0f, 0, NULL, NULL};

    static struct option LongOptions[] = {{"filter", required_argument, 0, 'f'}, {"scale", required_argument, 0, 's'},
                                           {"bias", required_argument, 0, 'b'},   {"verbose", no_argument, 0, 'v'},
                                           {"help", no_argument, 0, 'h'},         {0, 0, 0, 0}};

    int OptionIndex = 0;
    int C;

    while ((C = getopt_long(Argc, Argv, "f:s:b:vh", LongOptions, &OptionIndex)) != -1)
    {
        switch (C)
        {
        case 'f':
            Opts.Filter = ParseFilter(optarg);
            break;
        case 's':
            Opts.Scale = atof(optarg);
            break;
        case 'b':
            Opts.Bias = atof(optarg);
            break;
        case 'v':
            Opts.Verbose = 1;
            break;
        case 'h':
            PrintUsage(Argv[0]);
            exit(EXIT_SUCCESS);
        case '?':
            exit(EXIT_FAILURE);
        default:
            abort();
        }
    }

    if (optind + 2 != Argc)
    {
        fprintf(stderr, "Error: INPUT_FILE and OUTPUT_FILE are required\n");
        PrintUsage(Argv[0]);
        exit(EXIT_FAILURE);
    }

    Opts.InputFile = Argv[optind];
    Opts.OutputFile = Argv[optind + 1];

    return Opts;
}

static float *GetKernel(filter_type Filter)
{
    switch (Filter)
    {
    case FILTER_EDGE:
        return EdgeKernel;
    case FILTER_SHARPEN:
        return SharpenKernel;
    case FILTER_EMBOSS:
        return EmbossKernel;
    case FILTER_BLUR:
        return BlurKernel;
    case FILTER_IDENTITY:
        return IdentityKernel;
    default:
        return IdentityKernel;
    }
}

static const char *GetFilterName(filter_type Filter)
{
    switch (Filter)
    {
    case FILTER_EDGE:
        return "edge";
    case FILTER_SHARPEN:
        return "sharpen";
    case FILTER_EMBOSS:
        return "emboss";
    case FILTER_BLUR:
        return "blur";
    case FILTER_IDENTITY:
        return "identity";
    default:
        return "unknown";
    }
}

int main(int argc, char *argv[])
{
    options Opts = ParseCommandLine(argc, argv);

    struct timespec StartTime, EndTime;
    clock_gettime(CLOCK_MONOTONIC, &StartTime);

    // Load input image
    int Width, Height, Channels;
    unsigned char *HInput = stbi_load(Opts.InputFile, &Width, &Height, &Channels, 0);

    if (!HInput)
    {
        fprintf(stderr, "Error: Could not load image '%s'\n", Opts.InputFile);
        exit(EXIT_FAILURE);
    }

    if (Opts.Verbose)
    {
        printf("Loaded image: %dx%d, %d channels\n", Width, Height, Channels);
        printf("Filter: %s (scale=%.2f, bias=%.2f)\n", GetFilterName(Opts.Filter), Opts.Scale, Opts.Bias);
    }

    // Allocate host output
    size_t ImageSize = Width * Height * Channels;
    unsigned char *HOutput = (unsigned char *)malloc(ImageSize);
    if (!HOutput)
    {
        fprintf(stderr, "Error: Could not allocate output buffer\n");
        stbi_image_free(HInput);
        exit(EXIT_FAILURE);
    }

    // Allocate device memory
    unsigned char *DInput, *DOutput;
    cudaMalloc(&DInput, ImageSize);
    cudaMalloc(&DOutput, ImageSize);

    // Copy input to device
    cudaMemcpy(DInput, HInput, ImageSize, cudaMemcpyHostToDevice);

    struct timespec ProcessStart;
    clock_gettime(CLOCK_MONOTONIC, &ProcessStart);

    // Apply convolution
    float *Kernel = GetKernel(Opts.Filter);
    LaunchConvolution(DInput, DOutput, Width, Height, Channels, Kernel, 3, Opts.Scale, Opts.Bias);

    struct timespec ProcessEnd;
    clock_gettime(CLOCK_MONOTONIC, &ProcessEnd);

    // Copy result back to host
    cudaMemcpy(HOutput, DOutput, ImageSize, cudaMemcpyDeviceToHost);

    // Save output image
    int Result;
    if (strstr(Opts.OutputFile, ".png") || strstr(Opts.OutputFile, ".PNG"))
    {
        Result = stbi_write_png(Opts.OutputFile, Width, Height, Channels, HOutput, Width * Channels);
    }
    else if (strstr(Opts.OutputFile, ".jpg") || strstr(Opts.OutputFile, ".jpeg") ||
             strstr(Opts.OutputFile, ".JPG") || strstr(Opts.OutputFile, ".JPEG"))
    {
        Result = stbi_write_jpg(Opts.OutputFile, Width, Height, Channels, HOutput, 90);
    }
    else
    {
        Result = stbi_write_png(Opts.OutputFile, Width, Height, Channels, HOutput, Width * Channels);
    }

    if (!Result)
    {
        fprintf(stderr, "Error: Could not save image '%s'\n", Opts.OutputFile);
        exit(EXIT_FAILURE);
    }

    clock_gettime(CLOCK_MONOTONIC, &EndTime);

    if (Opts.Verbose)
    {
        double TotalTime = (EndTime.tv_sec - StartTime.tv_sec) + (EndTime.tv_nsec - StartTime.tv_nsec) / 1e9;
        double ProcessTime =
            (ProcessEnd.tv_sec - ProcessStart.tv_sec) + (ProcessEnd.tv_nsec - ProcessStart.tv_nsec) / 1e9;
        double DataMb = (ImageSize * 2) / (1024.0 * 1024.0);

        printf("Processing time  : %.6f sec\n", ProcessTime);
        printf("Total time       : %.6f sec\n", TotalTime);
        printf("Throughput       : %.2f MB/s\n", DataMb / ProcessTime);
        printf("Image saved to   : %s\n", Opts.OutputFile);
    }

    // Cleanup
    stbi_image_free(HInput);
    free(HOutput);
    cudaFree(DInput);
    cudaFree(DOutput);

    return 0;
}
