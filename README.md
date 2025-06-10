## 100-days-of-cuda

100 days of cuda challenge.

### Summary

This is a [challenge](https://github.com/hkproj/100-days-of-gpu/blob/main/CUDA.md) for writing a CUDA Kernel everyday for 100 days.

### Log

#### Day 01

Wrote a simple vector addition kernel.

#### Day 02

Wrote a simple matmul kernel. Only tested it with a simple 256x256 square matrix.

#### Day 03

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

#### Day 05

Wrote a image blur image converter based on example in PMPP.

Uses a simple average of surrounding pixels.

Only handles 1 channel images.

#### Day 06

Wrote a simple reverse array kernel. Goal was to understand
better how the `__syncthreads` directive works.

#### Day 07

Wrote a Tiled MatMul Kernel.  Finished chapter 4 and started chapter 5
from PMPP.  Wrote a simple program to query cuda device properties.

#### Day 08

Wrote a Random Matrix Generator using the cuRand library.
Values generated follow an uniform distribution over [0, 1].

Refactored structured of the project inspired by Computer Enhance
code.

Also created the following CLI's:

- matgen: Generate a sequence of matrices
- matmul: Multiplicates matrices from stdin

With them I was able to diagnose some bugs in the previous
matmul kernels.

#### Day 09

Wrote a Matrix Tranpose Kernel. Created the matpose CLI
that tranposes a matrix from the command line.

Read parts of Chapter 06 from PMPP.

Thinking I need to work on some instrumentation profiling.

### Notes

### Compiling directly to ptx

```
nvcc -ptx <input>.cu -o <output>.ptx
```

### Showing SASS results

```
nvcc -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 -cubin your_program.cu -o your_program.cubin
cuobjdump --dump-sass your_file.cubin > your_file.sass
```

For more details:

```
nvdisasm -g -c your_file.cubin > your_file.sass
```

### Resources:

- https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/
- https://kaixih.github.io/nvcc-options/
- [Cuda Occupancy Calculator](https://xmartlabs.github.io/cuda-calculator/)
