# Compilers & flags
CC := gcc
CUDA := nvcc
CFLAGS := -O3
CUDAFLAGS := -O -Wno-deprecated-gpu-targets

# Paths
BUILD_DIR := Build
SRC_DIR := Source

# Targets
TARGET_PART1 := lab4p1
TARGET_PART2 := lab4p2

TARGET_EXECUTABLES := \
	$(TARGET_PART1) \
	$(TARGET_PART2) \

# Objects and Dependencies
DEPS_PART2 := \
	Sobel.h \

OBJ_PART2 := \
	Sobel.o \


all: $(TARGET_EXECUTABLES)

$(TARGET_PART1):
	$(CUDA) -o $@ maxwell_griffin_$@.cu $(CUDAFLAGS)

$(TARGET_PART2): $(OBJ_PART2)
	$(CUDA) -c -o maxwell_griffin_$@.o maxwell_griffin_$@.cu $(CUDAFLAGS)
	$(CUDA) -o $@ maxwell_griffin_$@.o $^ $(CUDAFLAGS)

%.o: %.c $(DEPS_PART2)
	$(CUDA) -c -o $@ $< $(CUDAFLAGS)

.PHONY: clean
clean:
	@echo Cleaning build files...
	@rm -f *.o $(TARGET_EXECUTABLES)
