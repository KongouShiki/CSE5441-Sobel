# Compilers & flags
CUDA := nvcc
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

all: $(TARGET_EXECUTABLES)

$(TARGET_PART1):
	$(CUDA) -o $@ maxwell_griffin_$@.cu $(CUDAFLAGS)

$(TARGET_PART2):
	$(CUDA) -c -o maxwell_griffin_$@.o maxwell_griffin_$@.cu $(CUDAFLAGS)
	$(CUDA) -o $@ maxwell_griffin_$@.o $^ $(CUDAFLAGS)

.PHONY: clean
clean:
	@echo Cleaning build files...
	@rm -f $(TARGET_EXECUTABLES)
