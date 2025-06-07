build/main: main.cu build
	nvcc -DDEBUG_ENABLED=0 -g -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 main.cu -o ./build/main

build:
	mkdir -p build

run:
	./build/main

.PHONY: clean
clean:
	rm build/main
