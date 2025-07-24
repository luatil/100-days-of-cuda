################################
DEBUG_ENABLED = 1
################################

build/raytracer_dn: day_053_raytracer_003_main.cu
	nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 $< -o "./build/raytracer_dn"  -lcupti -lcuda

~/.local/bin/raytracer: build/raytracer_dn
	cp build/raytracer_dn ~/.local/bin/raytracer

raytracer: ~/.local/bin/raytracer
.PHONY: raytracer

temp.jpg: raytracer
	raytracer --width 800 --height 800

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
