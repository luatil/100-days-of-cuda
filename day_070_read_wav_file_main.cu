#include <stdio.h>
#include <stdint.h>

typedef uint32_t u32;
typedef uint16_t u16;
typedef int16_t s16;

struct RiffHeader {
  char ckID[4];
  u32  ckSize;
  char format[4];
};

struct WaveChunk {
  char chunkId[4];
  u32  chunkSize;
};

enum WaveFormatType : u16 {
  UNKNOWN,
  PCM,
  MS_ADPCM,
  IEEEFloatingPoint,
  ALAW = 6,
  MULAW,
  IMA_ADPCM  = 0x11,
  GSM610     = 0x31,
  MPEG       = 0x50,
  MPEGLAYER3 = 0x55,
};

struct WaveFormat {
  WaveFormatType formatTag;
  u16            channels;
  u32            samplesPerSec; //  [[comment("Sample Frequency")]];
  u32            avgBytesPerSec; // [[comment("BPS - Used to estimate buffer size")]];
  u16            blockAlign;
};


// Just divides each sample by 2
__global__ void MakeItQuieter(s16 *X, u32 N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (Tid < N)
    {
	X[Tid] /= 2;
    }
}

int main()
{
    RiffHeader Header = {};

    FILE *F = fopen("data/jfw.wav", "rb");

    fread(&Header, sizeof(Header), 1, F);

    printf("%.4s\n", Header.ckID);
    printf("%u\n", Header.ckSize);
    printf("%.4s\n", Header.format);

    WaveChunk Chunk = {};
    
    fread(&Chunk, sizeof(WaveChunk), 1, F);

    printf("%.4s\n", Chunk.chunkId);
    printf("%u\n", Chunk.chunkSize);

    WaveFormat Format = {};

    fread(&Format, sizeof(Format), 1, F);

    printf("[CHANNELS]: %d\n", Format.channels);
    printf("[SAMPLES_PER_SEC]: %d\n", Format.samplesPerSec);
    printf("[AVG_BYTES_PER_SEC]: %d\n", Format.avgBytesPerSec);
    printf("[BLOCK_ALIGNMENT]: %d\n", Format.blockAlign);

    WaveChunk OtherChunk = {};

    fread(&OtherChunk, sizeof(WaveChunk), 1, F);

    printf("%.4s\n", OtherChunk.chunkId);
    printf("%u\n", OtherChunk.chunkSize);

    char ListType[4];

    fread(ListType, sizeof(char), 4, F);
    printf("%.4s\n", ListType);

    WaveChunk Chunks[2] = {};

    fread(&Chunks[0], sizeof(WaveChunk), 1, F);

    printf("%.4s\n", Chunks[0].chunkId);
    printf("%u\n", Chunks[0].chunkSize);

    char ISFT[14];
    fread(ISFT, sizeof(char), 14, F);
    printf("%.14s\n", ISFT);

    fread(&Chunks[1], sizeof(WaveChunk), 1, F);

    printf("%.4s\n", Chunks[1].chunkId);
    printf("%u\n", Chunks[1].chunkSize);

    s16 *Data = (s16*)malloc(Chunks[1].chunkSize);
    fread(Data, Chunks[1].chunkSize, 1, F);

    fclose(F);

    s16 *GPUBuffer;
    cudaMalloc(&GPUBuffer, Chunks[1].chunkSize);
    cudaMemcpy(GPUBuffer, Data, Chunks[1].chunkSize, cudaMemcpyHostToDevice);

    const u32 NumberOfElements = Chunks[1].chunkSize / 2; // given s16
    const u32 BlockDim = 256;
    const u32 GridDim = (NumberOfElements + BlockDim - 1) / BlockDim;

    MakeItQuieter<<<GridDim, BlockDim>>>(GPUBuffer, NumberOfElements);

    cudaMemcpy(Data, GPUBuffer, Chunks[1].chunkSize, cudaMemcpyDeviceToHost);
    cudaFree(GPUBuffer);

    F = fopen("data/div2.s16", "wb");

    fwrite(Data, Chunks[1].chunkSize, 1, F);

    fclose(F);

    F = fopen("data/modified.wav", "wb");

    fwrite(&Header, sizeof(Header), 1, F);
    fwrite(&Chunk, sizeof(WaveChunk), 1, F);
    fwrite(&Format, sizeof(Format), 1, F);
    fwrite(&OtherChunk, sizeof(WaveChunk), 1, F);
    fwrite(ListType, sizeof(char), 4, F);
    fwrite(&Chunks[0], sizeof(WaveChunk), 1, F);
    fwrite(ISFT, sizeof(char), 14, F);
    fwrite(&Chunks[1], sizeof(WaveChunk), 1, F);
    fwrite(Data, Chunks[1].chunkSize, 1, F);

    fclose(F);
}
