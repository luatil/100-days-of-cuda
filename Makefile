################################
DEBUG_ENABLED = 1
################################


build/main: main.cu build
	nvcc -DDEBUG_ENABLED=1 -g -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 main.cu -o ./build/main

build/query_device_properties: query_device_properties.cu build
	nvcc -DDEBUG_ENABLED=1 -g -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 query_device_properties.cu -o ./build/query_device_properties

build:
	mkdir -p build

run: build/main
	./build/main

.PHONY: query_device_properties
query_device_properties: build/query_device_properties
	./build/query_device_properties


.PHONY: clean
clean:
	rm build/main
