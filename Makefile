################################
DEBUG_ENABLED = 1
################################

# all: view
all: result.txt

build/raytracer_dn: day_056_raytracer_006_main.cu
	nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 $< -o "./build/raytracer_dn"  -lcupti -lcuda

build/day_057_dn: day_057_add_main.cu
	nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 $< -o $@  -lcupti -lcuda

build/convolution_dn: day_058_convolution.cu
	nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 $< -o $@  -lcupti -lcuda

build/tensor_test_dn: day_060_tensor_lib_test_main.cu
	nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 $< -o $@  -lcupti -lcuda

build/mandelbrot_cli_dn: day_061_mandelbrot_cli.cu
	nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 $< -o $@  -lcupti -lcuda

result.txt: build/mandelbrot_cli_dn
	build/mandelbrot_cli_dn

test: result.txt
	diff result.txt goal.txt

~/.local/bin/raytracer: build/raytracer_dn
	cp build/raytracer_dn ~/.local/bin/raytracer

raytracer: ~/.local/bin/raytracer
.PHONY: raytracer

temp.jpg: raytracer
	raytracer --width 1024 --height 1024

view: temp.jpg
	xdg-open temp.jpg
.PHONY: view


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
