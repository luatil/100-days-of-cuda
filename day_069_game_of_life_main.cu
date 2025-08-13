#include <SDL2/SDL.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

// Initial grid dimensions
const int INITIAL_GRID_WIDTH = 400;
const int INITIAL_GRID_HEIGHT = 400;
const int CELL_SIZE = 2; // 1 pixel per cell

// Initial window dimensions
const int INITIAL_WINDOW_WIDTH = INITIAL_GRID_WIDTH * CELL_SIZE;
const int INITIAL_WINDOW_HEIGHT = INITIAL_GRID_HEIGHT * CELL_SIZE;

// CUDA kernel for Game of Life computation
__global__ void gameOfLifeKernel(unsigned char *current, unsigned char *next, int width, int height)
{
    // Get thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (x >= width || y >= height)
        return;

    int index = y * width + x;

    // Count live neighbors
    int neighbors = 0;
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0)
                continue; // Skip center cell

            int nx = x + dx;
            int ny = y + dy;

            // Handle boundaries (wrap around or treat as dead)
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                if (current[ny * width + nx] > 0)
                {
                    neighbors++;
                }
            }
        }
    }

    // Apply Game of Life rules
    bool isAlive = current[index] > 0;
    bool nextState = false;

    if (isAlive)
    {
        // Live cell survives with 2 or 3 neighbors
        nextState = (neighbors == 2 || neighbors == 3);
    }
    else
    {
        // Dead cell becomes alive with exactly 3 neighbors
        nextState = (neighbors == 3);
    }

    next[index] = nextState ? 255 : 0;
}

class GameOfLife
{
  private:
    // Host data
    unsigned char *h_currentGrid;
    unsigned char *h_nextGrid;

    // Device data
    unsigned char *d_currentGrid;
    unsigned char *d_nextGrid;

    // SDL components
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *texture;

    // Dynamic dimensions
    int gridWidth;
    int gridHeight;
    int windowWidth;
    int windowHeight;
    size_t gridSize;
    bool running;
    bool paused;

  public:
    GameOfLife() : window(nullptr), renderer(nullptr), texture(nullptr), running(true), paused(false)
    {
        // Initialize with default dimensions
        gridWidth = INITIAL_GRID_WIDTH;
        gridHeight = INITIAL_GRID_HEIGHT;
        windowWidth = INITIAL_WINDOW_WIDTH;
        windowHeight = INITIAL_WINDOW_HEIGHT;

        allocateMemory();
        initializeGrid();
        initializeSDL();
    }

    ~GameOfLife()
    {
        cleanup();
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    void allocateMemory()
    {
        gridSize = gridWidth * gridHeight * sizeof(unsigned char);

        // Allocate host memory
        h_currentGrid = new unsigned char[gridWidth * gridHeight];
        h_nextGrid = new unsigned char[gridWidth * gridHeight];

        // Allocate device memory
        cudaMalloc(&d_currentGrid, gridSize);
        cudaMalloc(&d_nextGrid, gridSize);
    }

    void cleanup()
    {
        if (h_currentGrid)
        {
            delete[] h_currentGrid;
            h_currentGrid = nullptr;
        }
        if (h_nextGrid)
        {
            delete[] h_nextGrid;
            h_nextGrid = nullptr;
        }
        if (d_currentGrid)
        {
            cudaFree(d_currentGrid);
            d_currentGrid = nullptr;
        }
        if (d_nextGrid)
        {
            cudaFree(d_nextGrid);
            d_nextGrid = nullptr;
        }
    }

    void initializeSDL()
    {
        if (SDL_Init(SDL_INIT_VIDEO) < 0)
        {
            std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
            exit(1);
        }

        window = SDL_CreateWindow("CUDA Game of Life", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, windowWidth,
                                  windowHeight, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

        if (!window)
        {
            std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
            exit(1);
        }

        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer)
        {
            std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
            exit(1);
        }

        texture =
            SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, gridWidth, gridHeight);

        if (!texture)
        {
            std::cerr << "Texture creation failed: " << SDL_GetError() << std::endl;
            exit(1);
        }
    }

    void initializeGrid()
    {
        // Initialize with random pattern
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 4);

        for (int i = 0; i < gridWidth * gridHeight; i++)
        {
            h_currentGrid[i] = (dis(gen) == 0) ? 255 : 0; // 20% chance of being alive
        }

        // Copy initial state to GPU
        cudaMemcpy(d_currentGrid, h_currentGrid, gridSize, cudaMemcpyHostToDevice);
    }

    void handleEvents()
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_WINDOWEVENT:
                if (event.window.event == SDL_WINDOWEVENT_RESIZED)
                {
                    handleResize(event.window.data1, event.window.data2);
                }
                break;
            case SDL_KEYDOWN:
                if (event.key.keysym.sym == SDLK_SPACE)
                {
                    paused = !paused;
                }
                else if (event.key.keysym.sym == SDLK_r)
                {
                    // Reset grid
                    initializeGrid();
                }
                else if (event.key.keysym.sym == SDLK_ESCAPE)
                {
                    running = false;
                }
                else if (event.key.keysym.sym == SDLK_q)
                {
		    running = false;
                }
                break;
            case SDL_MOUSEBUTTONDOWN:
                // Handle mouse clicks to toggle cells
                handleMouseClick(event.button.x, event.button.y);
                break;
            }
        }
    }

    void handleResize(int newWidth, int newHeight)
    {
        // Update window dimensions
        windowWidth = newWidth;
        windowHeight = newHeight;

        // Calculate new grid dimensions
        int newGridWidth = newWidth / CELL_SIZE;
        int newGridHeight = newHeight / CELL_SIZE;

        // Only resize if dimensions actually changed
        if (newGridWidth != gridWidth || newGridHeight != gridHeight)
        {
            gridWidth = newGridWidth;
            gridHeight = newGridHeight;

            // Clean up old memory
            cleanup();

            // Allocate new memory
            allocateMemory();

            // Reinitialize grid
            initializeGrid();

            // Recreate texture
            if (texture)
            {
                SDL_DestroyTexture(texture);
            }
            texture =
                SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, gridWidth, gridHeight);
        }
    }

    void handleMouseClick(int x, int y)
    {
        int cellX = x / CELL_SIZE;
        int cellY = y / CELL_SIZE;

        if (cellX >= 0 && cellX < gridWidth && cellY >= 0 && cellY < gridHeight)
        {
            int index = cellY * gridWidth + cellX;
            h_currentGrid[index] = h_currentGrid[index] ? 0 : 255;
            cudaMemcpy(d_currentGrid, h_currentGrid, gridSize, cudaMemcpyHostToDevice);
        }
    }

    void update()
    {
        // Launch CUDA kernel to compute the next generation
        dim3 blockSize(16, 16);
        dim3 gridDim((gridWidth + blockSize.x - 1) / blockSize.x, (gridHeight + blockSize.y - 1) / blockSize.y);
        gameOfLifeKernel<<<gridDim, blockSize>>>(d_currentGrid, d_nextGrid, gridWidth, gridHeight);
        cudaDeviceSynchronize();

        // Swap grids
        unsigned char *temp = d_currentGrid;
        d_currentGrid = d_nextGrid;
        d_nextGrid = temp;
    }

    void render()
    {
        // Copy current grid from GPU to CPU
        cudaMemcpy(h_currentGrid, d_currentGrid, gridSize, cudaMemcpyDeviceToHost);

        // Convert to RGB format for SDL
        unsigned char *pixels;
        int pitch;

        SDL_LockTexture(texture, nullptr, (void **)&pixels, &pitch);

        for (int y = 0; y < gridHeight; y++)
        {
            for (int x = 0; x < gridWidth; x++)
            {
                int gridIndex = y * gridWidth + x;
                int pixelIndex = y * pitch + x * 3;

                unsigned char value = h_currentGrid[gridIndex];
                pixels[pixelIndex] = value;     // R
                pixels[pixelIndex + 1] = value; // G
                pixels[pixelIndex + 2] = value; // B
            }
        }

        SDL_UnlockTexture(texture);

        // Render to screen
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    void run()
    {
        while (running)
        {
            handleEvents();
            if (!paused)
            {
                update();
            }
            render();

            // Control frame rate (roughly 10 FPS for slower animation)
            SDL_Delay(16);
        }
    }
};

int main()
{
    GameOfLife game;
    game.run();
    return 0;
}
