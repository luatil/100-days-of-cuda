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

#### Day 10

Wrote a Matrix/Vector Sum Kernel. It uses a tree-based
reduction to calculate the sum of all the elements in the
Matrix/Vector.

Created the matsum CLI.

#### Day 11

Worked on using the cuda event api to benchmark simple kernels.
Created examples on tuning kernel size, calculating effective
bandwidth and compute throughput.

#### Day 12

Created an utility to simplify calculating bandwidth,
timings and compute throughput of different Cuda Kernels.

Measured vector addition and matmul [simple and tiled]
with this new profiling harness.

#### Day 13

Worked with coarsening in vector add kernels. Did not
see any difference with a coarse factor of 2.

Worked with cudaOccupancyMaxPotentialBlockSize to find
likely optimal blockSizes.

Used the ncu CLI tool to check different metrics from
simple kernels.

References:

- https://www.youtube.com/watch?v=SGhfUhlowB4

#### Day 14

Wrote a repetition tester for profiling different kernels.
Profiled different vector addition kernels with
the repetition tester.

#### Day 15

Worked through the different reduction kernels in chapter 10
from PMPP.

#### Day 16

Profiled reduction kernels with different coarsening factors.
Started stuyding the cupti api for querying different
gpu counters.

Want to to modify the profiling harness to calculate
compute throughput based on actual inst\_executed
rather than a theoretical Arithmetic Intensity.

Best resource I found for the api is
https://github.com/eunomia-bpf/cupti-tutorial

Was able to compile, but getting errors
that this api is not supported on my gpu (3060).

#### Day 17

Wrote a softmax kernel. Studied llm.c implementation
of softmax kernels. Re-watched Simon Oz video about
it. Implemented a cpu reference kernel and a naive
single block gpu kernel with shared memory.

Want to profile both of them before implementing
more perfomant versions.

#### Day 18

WIP better softmax impl.

#### Day 19

Finished implementation of a better softmax kernel.
Used atomicCAS which I am not sure if it is a really
good idea.

Kinda need to profile the kernels to check the
overall perf.

#### Day 20

Created kernels for computing the Dot Product
of two vectors. Based on a LeetGPU problem.

Need to find a better way of organizing
multiple different kernels that fit the same
function definition.

Also need to find a way to tune the kernels
with BLOCK\_DIM and COARSE\_FACTOR that I
should know at compile time.

#### Day 21

Implemented a MSE Kernel.


#### Day 22

Implemented some PrefixSum kernels.

This video helped a lot:

- https://www.youtube.com/watch?v=ZKrWyEqqPVY

#### Day 23

Implemented a Histogram kernel.

#### Day 24

WIP Topk Kernel

#### Day 25

Reviewing Prefix Sum.

#### Day 26

Profiling histogram kernel

#### Day 27

Working with Cuda Graphs API.

#### Day 28

Fixing a bug in the Cuda Graph Example

#### Day 29

Leaky Relu kernel

#### Day 30

WIP Softmax Attention.

#### Day 31

Revisiting / Optimizing Histogram Kernel

#### Day 32

WIP on categorical cross entropy loss kernel

#### Day 33

More WIP on categorical cross entropy loss kernel

#### Day 34

Revisiting Monte Carlo Integration with a better kernel design

#### Day 35

Revisiting Reduction Kernel

#### Day 36

Solved the Password Cracking (FNV-1a) LeetGPU problem.

#### Day 37

Solved the GEMM (FP16) LeetGPU problem.

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

### Ideas

Some future ideas for unix-like utils:

Matrix ones:

- mat2img: Receives a matrix from a process and outputs a img. For instance: matgen -h 20 -w 20 -n 1 | mat2img
- matshape: Receives a matrix from a certain format and converts it to another one. For instance: matgen -h 3 -w 4 | matshape -h 4 -w 3

Image ones:

I think that I could leverage something like the kitty protocol to allow for cool image composition things:

- Read cool_image.jpg and converts it with imblur: imread cool_image.jpg | imblur
- Perform edge detection after blurring the image: imread cool_image.jpg | imblur --radius 10 | imedge

LLM like ones:

It would be awesome if we could run an LLM by only using unix pipes.

cat temp.txt | tknize --bpe | embed --rotary | ... 

And so on.

Refactoring matmul:

{ matgen -h 3 -w 4; matgen -h 4 -w 10; } | matmul

Possible optimizations:

This pipeline style of thing could even be lazy, otherwise floating
point parsing will very clearly be a bottleneck.  Although we could do
floating point parsing in the GPU for funsies.  I could also just send
raw data over the pipes. And use matprint for converting to a normal
format. e.g. matgen --height 3 --width 4 | matprint

Compilation:

Considering switching to something like Cmake. This find_cuda_lib looks
kinda of annoying to do otherwise.

I also want to only use nvcc for compiling the .cu files and use a normal
compiler for the other things. Which could be easier to do with cmake (maybe?).

I kinda want to be able to do something like:

- matgen -w 3 -h 4 --cpu 
- matgen -w 3 -h 4 --gpu

To force the execution.

Default if available would go to the GPU otherwise would go to the CPU.

Would very much like to have a static binary that could easily be copied
between systems that have or don't have GPUS.

This could double down as a sort of learning path for SIMD on the CPU.


### Resources:

- https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/
- https://kaixih.github.io/nvcc-options/
- [Cuda Occupancy Calculator](https://xmartlabs.github.io/cuda-calculator/)
