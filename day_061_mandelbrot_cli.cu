#include "day_003_vendor_libraries.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct MandelbrotParams
{
    int width;
    int height;
    double centerX;
    double centerY;
    double zoom;
    int maxIter;
    char outputFile[256];
};

__global__ void MandelbrotKernel(int *output, int width, int height, double centerX, double centerY, double zoom,
                                 int maxIter)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height)
        return;

    double scale = 4.0 / (zoom * min(width, height));
    double x0 = centerX + (col - width / 2.0) * scale;
    double y0 = centerY + (row - height / 2.0) * scale;

    double x = 0.0;
    double y = 0.0;
    int iter = 0;

    while (x * x + y * y <= 4.0 && iter < maxIter)
    {
        double xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        iter++;
    }

    output[row * width + col] = iter;
}

void PrintUsage(const char *progName)
{
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -w, --width <int>     Width in pixels (default: 1024)\n");
    printf("  -h, --height <int>    Height in pixels (default: 1024)\n");
    printf("  -x, --centerX <float> Center X coordinate (default: -0.5)\n");
    printf("  -y, --centerY <float> Center Y coordinate (default: 0.0)\n");
    printf("  -z, --zoom <float>    Zoom level (default: 1.0)\n");
    printf("  -i, --iterations <int> Max iterations (default: 100)\n");
    printf("  -o, --output <file>   Output JPEG filename (default: mandelbrot.jpg)\n");
    printf("  --help                Show this help message\n");
}

MandelbrotParams ParseArgs(int argc, char **argv)
{
    MandelbrotParams params = {
        .width = 1024, .height = 1024, .centerX = -0.5, .centerY = 0.0, .zoom = 1.0, .maxIter = 100, .outputFile = ""};
    strcpy(params.outputFile, "mandelbrot.jpg");

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--width") == 0)
        {
            if (i + 1 < argc)
                params.width = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--height") == 0)
        {
            if (i + 1 < argc)
                params.height = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-x") == 0 || strcmp(argv[i], "--centerX") == 0)
        {
            if (i + 1 < argc)
                params.centerX = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-y") == 0 || strcmp(argv[i], "--centerY") == 0)
        {
            if (i + 1 < argc)
                params.centerY = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-z") == 0 || strcmp(argv[i], "--zoom") == 0)
        {
            if (i + 1 < argc)
                params.zoom = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0)
        {
            if (i + 1 < argc)
                params.maxIter = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0)
        {
            if (i + 1 < argc)
                strcpy(params.outputFile, argv[++i]);
        }
        else if (strcmp(argv[i], "--help") == 0)
        {
            PrintUsage(argv[0]);
            exit(0);
        }
    }

    return params;
}

void SaveJPEG(int *hostData, int width, int height, int maxIter, const char *filename)
{
    unsigned char *imageData = (unsigned char *)malloc(width * height * 3);

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int iter = hostData[row * width + col];
            int idx = (row * width + col) * 3;

            unsigned char r, g, b;
            if (iter == maxIter)
            {
                r = g = b = 0;
            }
            else
            {
                double t = (double)iter / maxIter;
                r = (unsigned char)(255 * (0.5 + 0.5 * cos(3.0 + t * 12.0)));
                g = (unsigned char)(255 * (0.5 + 0.5 * cos(2.0 + t * 12.0)));
                b = (unsigned char)(255 * (0.5 + 0.5 * cos(1.0 + t * 12.0)));
            }

            imageData[idx] = r;
            imageData[idx + 1] = g;
            imageData[idx + 2] = b;
        }
    }

    int result = stbi_write_jpg(filename, width, height, 3, imageData, 90);
    if (result)
    {
        printf("Saved JPEG image to: %s\n", filename);
    }
    else
    {
        printf("Error: Could not save image to %s\n", filename);
    }

    free(imageData);
}

int main(int argc, char **argv)
{
    MandelbrotParams params = ParseArgs(argc, argv);

    printf("Generating Mandelbrot set...\n");
    printf("Resolution: %dx%d, Center: (%.6f, %.6f), Zoom: %.2f, Iterations: %d\n", params.width, params.height,
           params.centerX, params.centerY, params.zoom, params.maxIter);
    printf("Output file: %s\n", params.outputFile);

    int *deviceData;
    int *hostData = (int *)malloc(params.width * params.height * sizeof(int));

    cudaMalloc(&deviceData, params.width * params.height * sizeof(int));

    dim3 blockSize(16, 16);
    dim3 gridSize((params.width + blockSize.x - 1) / blockSize.x, (params.height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MandelbrotKernel<<<gridSize, blockSize>>>(deviceData, params.width, params.height, params.centerX, params.centerY,
                                              params.zoom, params.maxIter);
    cudaEventRecord(stop);

    cudaMemcpy(hostData, deviceData, params.width * params.height * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    SaveJPEG(hostData, params.width, params.height, params.maxIter, params.outputFile);

    printf("\nGeneration time: %.2f ms\n", milliseconds);

    free(hostData);
    cudaFree(deviceData);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
