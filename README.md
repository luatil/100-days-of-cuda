# 100-days-of-cuda

100 days of cuda challenge.

## Summary

This is a [challenge](https://github.com/hkproj/100-days-of-gpu/blob/main/CUDA.md) for writing a CUDA Kernel everyday for 100 days.

## Log

### Day 01

Wrote a simple vector addition kernel.

### Day 02

Wrote a simple matmul kernel. Only tested it with a simple 256x256 square matrix.

### Day 03

Wrote a color inversion kernel. Implemented simple CLI program that
reads input and output filenames and inverts the input image to the
output image.

Based on LeetGPU Color Inversion challenge.

Uses `stb_image.h` and `stb_image_write.h` for image reading and writing.

## Notes

## Compiling directly to ptx

```
nvcc -ptx <input>.cu -o <output>.ptx
```

## Showing SASS results

```
nvcc -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 -cubin your_program.cu -o your_program.cubin
cuobjdump --dump-sass your_file.cubin > your_file.sass
```

For more details:

```
nvdisasm -g -c your_file.cubin > your_file.sass
```

## Resources:

- https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/
- https://kaixih.github.io/nvcc-options/

