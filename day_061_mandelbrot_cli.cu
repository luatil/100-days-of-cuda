#include "day_003_vendor_libraries.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mandelbrot_params
{
    int Width;
    int Height;
    double CenterX;
    double CenterY;
    double Zoom;
    int MaxIter;
    char OutputFile[256];
};

__global__ void MandelbrotKernel(int *Output, int Width, int Height, double CenterX, double CenterY, double Zoom,
                                 int MaxIter)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col >= Width || Row >= Height)
        return;

    double Scale = 4.0 / (Zoom * min(Width, Height));
    double X0 = CenterX + (Col - Width / 2.0) * Scale;
    double Y0 = CenterY + (Row - Height / 2.0) * Scale;

    double X = 0.0;
    double Y = 0.0;
    int Iter = 0;

    while (X * X + Y * Y <= 4.0 && Iter < MaxIter)
    {
        double Xtemp = X * X - Y * Y + X0;
        Y = 2 * X * Y + Y0;
        X = Xtemp;
        Iter++;
    }

    Output[Row * Width + Col] = Iter;
}

void PrintUsage(const char *ProgName)
{
    printf("Usage: %s [options]\n", ProgName);
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

mandelbrot_params ParseArgs(int Argc, char **Argv)
{
    mandelbrot_params Params = {
        .Width = 1024, .Height = 1024, .CenterX = -0.5, .CenterY = 0.0, .Zoom = 1.0, .MaxIter = 100, .OutputFile = ""};
    strcpy(Params.OutputFile, "mandelbrot.jpg");

    for (int I = 1; I < Argc; I++)
    {
        if (strcmp(Argv[I], "-w") == 0 || strcmp(Argv[I], "--width") == 0)
        {
            if (I + 1 < Argc)
                Params.Width = atoi(Argv[++I]);
        }
        else if (strcmp(Argv[I], "-h") == 0 || strcmp(Argv[I], "--height") == 0)
        {
            if (I + 1 < Argc)
                Params.Height = atoi(Argv[++I]);
        }
        else if (strcmp(Argv[I], "-x") == 0 || strcmp(Argv[I], "--centerX") == 0)
        {
            if (I + 1 < Argc)
                Params.CenterX = atof(Argv[++I]);
        }
        else if (strcmp(Argv[I], "-y") == 0 || strcmp(Argv[I], "--centerY") == 0)
        {
            if (I + 1 < Argc)
                Params.CenterY = atof(Argv[++I]);
        }
        else if (strcmp(Argv[I], "-z") == 0 || strcmp(Argv[I], "--zoom") == 0)
        {
            if (I + 1 < Argc)
                Params.Zoom = atof(Argv[++I]);
        }
        else if (strcmp(Argv[I], "-i") == 0 || strcmp(Argv[I], "--iterations") == 0)
        {
            if (I + 1 < Argc)
                Params.MaxIter = atoi(Argv[++I]);
        }
        else if (strcmp(Argv[I], "-o") == 0 || strcmp(Argv[I], "--output") == 0)
        {
            if (I + 1 < Argc)
                strcpy(Params.OutputFile, Argv[++I]);
        }
        else if (strcmp(Argv[I], "--help") == 0)
        {
            PrintUsage(Argv[0]);
            exit(0);
        }
    }

    return Params;
}

void SaveJPEG(int *HostData, int Width, int Height, int MaxIter, const char *Filename)
{
    unsigned char *ImageData = (unsigned char *)malloc(Width * Height * 3);

    for (int Row = 0; Row < Height; Row++)
    {
        for (int Col = 0; Col < Width; Col++)
        {
            int Iter = HostData[Row * Width + Col];
            int Idx = (Row * Width + Col) * 3;

            unsigned char R, G, B;
            if (Iter == MaxIter)
            {
                R = G = B = 0;
            }
            else
            {
                double T = (double)Iter / MaxIter;
                R = (unsigned char)(255 * (0.5 + 0.5 * cos(3.0 + T * 12.0)));
                G = (unsigned char)(255 * (0.5 + 0.5 * cos(2.0 + T * 12.0)));
                B = (unsigned char)(255 * (0.5 + 0.5 * cos(1.0 + T * 12.0)));
            }

            ImageData[Idx] = R;
            ImageData[Idx + 1] = G;
            ImageData[Idx + 2] = B;
        }
    }

    int Result = stbi_write_jpg(Filename, Width, Height, 3, ImageData, 90);
    if (Result)
    {
        printf("Saved JPEG image to: %s\n", Filename);
    }
    else
    {
        printf("Error: Could not save image to %s\n", Filename);
    }

    free(ImageData);
}

int main(int argc, char **argv)
{
    mandelbrot_params Params = ParseArgs(argc, argv);

    printf("Generating Mandelbrot set...\n");
    printf("Resolution: %dx%d, Center: (%.6f, %.6f), Zoom: %.2f, Iterations: %d\n", Params.Width, Params.Height,
           Params.CenterX, Params.CenterY, Params.Zoom, Params.MaxIter);
    printf("Output file: %s\n", Params.OutputFile);

    int *DeviceData;
    int *HostData = (int *)malloc(Params.Width * Params.Height * sizeof(int));

    cudaMalloc(&DeviceData, Params.Width * Params.Height * sizeof(int));

    dim3 BlockSize(16, 16);
    dim3 GridSize((Params.Width + BlockSize.x - 1) / BlockSize.x, (Params.Height + BlockSize.y - 1) / BlockSize.y);

    cudaEvent_t Start, Stop;
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);

    cudaEventRecord(Start);
    MandelbrotKernel<<<GridSize, BlockSize>>>(DeviceData, Params.Width, Params.Height, Params.CenterX, Params.CenterY,
                                              Params.Zoom, Params.MaxIter);
    cudaEventRecord(Stop);

    cudaMemcpy(HostData, DeviceData, Params.Width * Params.Height * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(Stop);
    float Milliseconds = 0;
    cudaEventElapsedTime(&Milliseconds, Start, Stop);

    SaveJPEG(HostData, Params.Width, Params.Height, Params.MaxIter, Params.OutputFile);

    printf("\nGeneration time: %.2f ms\n", Milliseconds);

    free(HostData);
    cudaFree(DeviceData);
    cudaEventDestroy(Start);
    cudaEventDestroy(Stop);

    return 0;
}
