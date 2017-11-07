# Compilers & flags
CC := gcc
CUDA := nvcc
CFLAGS := -O3
CUDAFLAGS := -O

# Paths
BUILD_DIR := Build
SRC_DIR := Source

# Targets
TARGET_PART1 := lab4p1
TARGET_PART2 := lab4p2

TARGET_EXECUTABLES := \
	$(TARGET_PART1)

# Objects
OBJ_PART1 := \


OBJ_PART2 := \
	bmpReader.o


%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all: $(TARGET_EXECUTABLES)

$(TARGET_PART1): $(OBJ)
	$(CUDA) -c -o maxwell_griffin_$@.o maxwell_griffin_$@.cu $(CUDAFLAGS)
	$(CC) -o $@ $^ $(CFLAGS)

$(TARGET_PART2): $(OBJ)
	$(CUDA) -c -o maxwell_griffin_$@.o maxwell_griffin_$@.cu $(CUDAFLAGS)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean
clean:
	@echo Cleaning build files...
	@rm -f *.o disposable persistent
