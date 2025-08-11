################################
DEBUG_ENABLED = 1
################################

# all: view
all: result.txt

.PHONY: all-main
all-main: $(patsubst %_main.cu,build/%_main,$(wildcard *_main.cu))

build/%_main: %_main.cu | build
	nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 $< -o $@  -lcupti -lcuda

build/%_test_dn: %_test.cu
	@nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 $< -o $@  -lcupti -lcuda

input.txt:
	seq 1 100000000 | shuf > input.txt

sorted.txt: build/day_068_something_main input.txt
	cat input.txt | time build/day_068_something_main > sorted.txt

result.txt: sorted.txt
	cat sorted.txt | issorted > result.txt

.PHONY: test
test: $(patsubst %_test.cu,build/%_test_dn,$(wildcard *_test.cu))
	@mkdir -p build
	@passed=0; failed=0; \
	for test in $^; do \
		if ./$$test; then \
			echo "✓ $$(basename $$test) PASSED"; \
			passed=$$((passed + 1)); \
		else \
			echo "✗ $$(basename $$test) FAILED"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo "Tests completed: $$passed passed, $$failed failed"


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


.PHONY: format
format:
	clang-format -i *.cu

.PHONY: format-names
format-names:
	clang-tidy --fix --fix-errors --checks=readability-identifier-naming *.cu *.h

.PHONY: clean
clean:
	rm build/main
