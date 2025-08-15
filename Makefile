################################
DEBUG_ENABLED = 1
################################

# all: view
all: result.txt

.PHONY: all-main
all-main: $(patsubst %_main.cu,build/%_main,$(wildcard *_main.cu))

# build/day_069_something_main: day_069_something_main.cu | build
build/day_069_game_of_life_main: day_069_game_of_life_main.cu | build
	nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 $< -o $@ -lSDL2 -lcupti -lcuda

.PHONY: run-day-069
run-day-069: build/day_069_game_of_life_main
	./build/day_069_game_of_life_main

build/%_main: %_main.cu | build
	nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 $< -o $@  -lcupti -lcuda

build/%_test_dn: %_test.cu
	@nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 $< -o $@  -lcupti -lcuda

result.txt: build/day_071_edge_detection_main
	./$< > result.txt

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

.PHONY: query_device_properties
query_device_properties: build/query_device_properties
	./build/query_device_properties


data/jfw.wav:
	curl -o $@ --location https://github.com/ggml-org/whisper.cpp/raw/refs/heads/master/samples/jfk.wav

.PHONY: format
format:
	clang-format -i *.cu

.PHONY: format-names
format-names:
	clang-tidy --fix --fix-errors --checks=readability-identifier-naming *.cu *.h

.PHONY: clean
clean:
	rm build/main
