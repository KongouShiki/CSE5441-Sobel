# Compilers & flags
CUDA := nvcc
CUDAFLAGS := -arch=sm_60 -O -Wno-deprecated-gpu-targets

# Paths
BUILD_DIR := Build
SRC_DIR := Source

# Targets
TARGET_PART1 := lab4p1
TARGET_PART2 := lab4p2

TARGET_EXECUTABLES := \
	$(TARGET_PART1) \
	$(TARGET_PART2) \

# Objects
OBJ_PART2 := \
	Sobel.o \
	Stencil.o \

LIB_OBJ_PART2 := \
	nvcc60_bmpReader.o \


all: $(TARGET_EXECUTABLES)

$(TARGET_PART1):
	$(CUDA) -o $@ maxwell_griffin_$@.cu $(CUDAFLAGS)

$(TARGET_PART2): $(OBJ_PART2)
	$(CUDA) -dc -o maxwell_griffin_$@.o maxwell_griffin_$@.cu $(CUDAFLAGS)
	$(CUDA) -o $@ maxwell_griffin_$@.o $(LIB_OBJ_PART2) $^ $(CUDAFLAGS)

%.o: %.c
	$(CUDA) -x c -std=c99 -c -o $@ $< $(CUDAFLAGS)

.PHONY: clean
clean:
	@echo Cleaning build files...
	@rm -f $(TARGET_EXECUTABLES)
