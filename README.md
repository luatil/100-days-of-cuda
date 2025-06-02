# 100-days-of-cuda
100 days of cuda challenge

## Summary

### Day 01

Wrote a simple vector addition kernel.

### Day 02

Wrote a simple matmul kernel. Only tested it with a simple 256x256 square matrix.

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

