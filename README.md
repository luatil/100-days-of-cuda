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

#### Day 04

Wrote a grayscale image converter based on example in PMPP. Implemented
a simple CLI that reads input and output filenames and writes the
grayscale image.

Uses the following formula for calculating grayscale values:

`L = 0.21*r + 0.72*g + 0.07b`


Uses `stb_image.h` and `stb_image_write.h` for image reading and writing.

Major difference from the book is handling RGBA images versus RGA ones
in the Kernel.

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

