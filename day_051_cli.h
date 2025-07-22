#ifndef DAY_051_CLI_H
#define DAY_051_CLI_H
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    int Width;
    int Height;
    char *OutputFilename;
    int CircleX;
    int CircleY;
    int Radius;
} options;

static void PrintUsage(const char *ProgramName)
{
    printf("Usage: %s [OPTION]...\n", ProgramName);
    printf("Renders a circle using CUDA and signed distance field (SDF) technique.\n");
    printf("Outputs the result as a JPEG image with anti-aliased edges.\n\n");
    printf("Options:\n");
    printf("  -w, --width=WIDTH          set image width in pixels (default: 400)\n");
    printf("  -h, --height=HEIGHT        set image height in pixels (default: 400)\n");
    printf("  -o, --output-filename=FILE output filename (default: temp.jpg)\n");
    printf("  -x, --cx=X                 circle center X coordinate (default: 200)\n");
    printf("  -y, --cy=Y                 circle center Y coordinate (default: 200)\n");
    printf("  -r, --radius=R             circle radius in pixels (default: 100)\n");
    printf("      --help                 display this help and exit\n");
}

static options ParseCommandLine(int ArgumentCount, char *Arguments[])
{
    options Opts = {400, 400, (char *)"temp.jpg", 200, 200, 100};

    static struct option LongOptions[] = {{"width", required_argument, 0, 'w'},
                                          {"height", required_argument, 0, 'h'},
                                          {"output-filename", required_argument, 0, 'o'},
                                          {"cx", required_argument, 0, 'x'},
                                          {"cy", required_argument, 0, 'y'},
                                          {"radius", required_argument, 0, 'r'},
                                          {"help", no_argument, 0, '?'},
                                          {0, 0, 0, 0}};

    int OptionIndex = 0;
    int Char;

    while ((Char = getopt_long(ArgumentCount, Arguments, "w:h:o:x:y:r:", LongOptions, &OptionIndex)) != -1)
    {
        switch (Char)
        {
        case 'w':
            Opts.Width = atoi(optarg);
            if (Opts.Width <= 0)
            {
                fprintf(stderr, "Error: Width must be positive\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'h':
            Opts.Height = atoi(optarg);
            if (Opts.Height <= 0)
            {
                fprintf(stderr, "Error: Height must be positive\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'o':
            Opts.OutputFilename = optarg;
            break;
        case 'x':
            Opts.CircleX = atoi(optarg);
            break;
        case 'y':
            Opts.CircleY = atoi(optarg);
            break;
        case 'r':
            Opts.Radius = atoi(optarg);
            if (Opts.Radius <= 0)
            {
                fprintf(stderr, "Error: Radius must be positive\n");
                exit(EXIT_FAILURE);
            }
            break;
        case '?':
        default:
            PrintUsage(Arguments[0]);
            exit(EXIT_SUCCESS);
            break;
        }
    }

    return Opts;
}
#endif
