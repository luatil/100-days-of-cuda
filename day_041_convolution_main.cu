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
} FilterType;

typedef struct
{
    FilterType filter;
    float scale;
    float bias;
    int verbose;
    char *input_file;
    char *output_file;
} Options;

// Predefined convolution kernels
float edge_kernel[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};

float sharpen_kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

float emboss_kernel[9] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};

float blur_kernel[9] = {1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9};

float identity_kernel[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};

static void PrintUsage(const char *program_name)
{
    printf("Usage: %s [OPTION]... INPUT_FILE OUTPUT_FILE\n", program_name);
    printf("Applies convolution filters to images using CUDA acceleration.\n\n");
    printf("Options:\n");
    printf("  -f, --filter=FILTER    filter to apply (edge, sharpen, emboss, blur, identity)\n");
    printf("  -s, --scale=SCALE      scaling factor (default: 1.0)\n");
    printf("  -b, --bias=BIAS        bias value (default: 0.0)\n");
    printf("  -v, --verbose          print timing information\n");
    printf("      --help             display this help and exit\n");
}

static FilterType ParseFilter(const char *filter_str)
{
    if (strcmp(filter_str, "edge") == 0)
        return FILTER_EDGE;
    if (strcmp(filter_str, "sharpen") == 0)
        return FILTER_SHARPEN;
    if (strcmp(filter_str, "emboss") == 0)
        return FILTER_EMBOSS;
    if (strcmp(filter_str, "blur") == 0)
        return FILTER_BLUR;
    if (strcmp(filter_str, "identity") == 0)
        return FILTER_IDENTITY;

    fprintf(stderr, "Error: Unknown filter '%s'\n", filter_str);
    fprintf(stderr, "Available filters: edge, sharpen, emboss, blur, identity\n");
    exit(EXIT_FAILURE);
}

static Options ParseCommandLine(int argc, char *argv[])
{
    Options opts = {FILTER_EDGE, 1.0f, 0.0f, 0, NULL, NULL};

    static struct option long_options[] = {{"filter", required_argument, 0, 'f'}, {"scale", required_argument, 0, 's'},
                                           {"bias", required_argument, 0, 'b'},   {"verbose", no_argument, 0, 'v'},
                                           {"help", no_argument, 0, 'h'},         {0, 0, 0, 0}};

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "f:s:b:vh", long_options, &option_index)) != -1)
    {
        switch (c)
        {
        case 'f':
            opts.filter = ParseFilter(optarg);
            break;
        case 's':
            opts.scale = atof(optarg);
            break;
        case 'b':
            opts.bias = atof(optarg);
            break;
        case 'v':
            opts.verbose = 1;
            break;
        case 'h':
            PrintUsage(argv[0]);
            exit(EXIT_SUCCESS);
        case '?':
            exit(EXIT_FAILURE);
        default:
            abort();
        }
    }

    if (optind + 2 != argc)
    {
        fprintf(stderr, "Error: INPUT_FILE and OUTPUT_FILE are required\n");
        PrintUsage(argv[0]);
        exit(EXIT_FAILURE);
    }

    opts.input_file = argv[optind];
    opts.output_file = argv[optind + 1];

    return opts;
}

static float *GetKernel(FilterType filter)
{
    switch (filter)
    {
    case FILTER_EDGE:
        return edge_kernel;
    case FILTER_SHARPEN:
        return sharpen_kernel;
    case FILTER_EMBOSS:
        return emboss_kernel;
    case FILTER_BLUR:
        return blur_kernel;
    case FILTER_IDENTITY:
        return identity_kernel;
    default:
        return identity_kernel;
    }
}

static const char *GetFilterName(FilterType filter)
{
    switch (filter)
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
    Options opts = ParseCommandLine(argc, argv);

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Load input image
    int width, height, channels;
    unsigned char *h_input = stbi_load(opts.input_file, &width, &height, &channels, 0);

    if (!h_input)
    {
        fprintf(stderr, "Error: Could not load image '%s'\n", opts.input_file);
        exit(EXIT_FAILURE);
    }

    if (opts.verbose)
    {
        printf("Loaded image: %dx%d, %d channels\n", width, height, channels);
        printf("Filter: %s (scale=%.2f, bias=%.2f)\n", GetFilterName(opts.filter), opts.scale, opts.bias);
    }

    // Allocate host output
    size_t image_size = width * height * channels;
    unsigned char *h_output = (unsigned char *)malloc(image_size);
    if (!h_output)
    {
        fprintf(stderr, "Error: Could not allocate output buffer\n");
        stbi_image_free(h_input);
        exit(EXIT_FAILURE);
    }

    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);

    struct timespec process_start;
    clock_gettime(CLOCK_MONOTONIC, &process_start);

    // Apply convolution
    float *kernel = GetKernel(opts.filter);
    LaunchConvolution(d_input, d_output, width, height, channels, kernel, 3, opts.scale, opts.bias);

    struct timespec process_end;
    clock_gettime(CLOCK_MONOTONIC, &process_end);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    // Save output image
    int result;
    if (strstr(opts.output_file, ".png") || strstr(opts.output_file, ".PNG"))
    {
        result = stbi_write_png(opts.output_file, width, height, channels, h_output, width * channels);
    }
    else if (strstr(opts.output_file, ".jpg") || strstr(opts.output_file, ".jpeg") ||
             strstr(opts.output_file, ".JPG") || strstr(opts.output_file, ".JPEG"))
    {
        result = stbi_write_jpg(opts.output_file, width, height, channels, h_output, 90);
    }
    else
    {
        result = stbi_write_png(opts.output_file, width, height, channels, h_output, width * channels);
    }

    if (!result)
    {
        fprintf(stderr, "Error: Could not save image '%s'\n", opts.output_file);
        exit(EXIT_FAILURE);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    if (opts.verbose)
    {
        double total_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        double process_time =
            (process_end.tv_sec - process_start.tv_sec) + (process_end.tv_nsec - process_start.tv_nsec) / 1e9;
        double data_mb = (image_size * 2) / (1024.0 * 1024.0);

        printf("Processing time  : %.6f sec\n", process_time);
        printf("Total time       : %.6f sec\n", total_time);
        printf("Throughput       : %.2f MB/s\n", data_mb / process_time);
        printf("Image saved to   : %s\n", opts.output_file);
    }

    // Cleanup
    stbi_image_free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
