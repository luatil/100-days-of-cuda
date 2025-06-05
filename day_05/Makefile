build/main: main.cu build
	nvcc -DDEBUG_ENABLED=0 -g -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 main.cu -o ./build/main

data/nvidia_blurred.jpg: build/main data/nvidia_cuda_logo.jpg
	./build/main data/nvidia_cuda_logo.jpg data/nvidia_blurred.jpg

build:
	mkdir -p build

.PHONY: clean
clean:
	rm build/main
