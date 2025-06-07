#!/bin/bash

mkdir -p build
nvcc -DDEBUG_ENABLED=0 -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 main.cu -o ./build/main
