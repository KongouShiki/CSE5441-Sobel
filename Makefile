# Compilers
CC := gcc
CUDA := nvcc

# Paths
BUILD_DIR := Build
SRC_DIR := Source
SRC_PART1_DIR := $(SRC_DIR)/Part1
SRC_PART2_DIR := $(SRC_DIR)/Part2

# Targets
TARGET_PART1 := lab4p1
TARGET_PART2 := lab4p2

TARGET_EXECUTABLES := \
	$(TARGET_PART1)

# File lists
SRCS := $(shell find $(SRC_DIR) -name *.c)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

CFLAGS := -O3
CUDAFLAGS := -O -Wno-deprecated-gpu-targets

all: $(TARGET_EXECUTABLES)

$(TARGET_PART1): $(OBJS)
	$(CUDA) $(CUDAFLAGS) -o $@ $(SRC_PART1_DIR)/maxwell_griffin_$@.cu

$(TARGET_PART2): $(OBJS)
	@mkdir -p $(BUILD_DIR)/$(SRC_PART2_DIR)
	$(CUDA) $(CUDAFLAGS) -c $(SRC_PART2_DIR)/maxwell_griffin_$@.cu -o $(BUILD_DIR)/$(SRC_PART2_DIR)/maxwell_griffin_$@.o
	$(CUDA) $(CUDAFLAGS) -o $@ $(OBJS) $(BUILD_DIR)/$(SRC_PART2_DIR)/maxwell_griffin_$@.o

# c source
$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(CUDA) $(CUDAFLAGS) -c $< -o $@


.PHONY: clean package test

clean:
	@echo Cleaning build files...
	@$(RM) -rf $(BUILD_DIR)
	@$(RM) $(TARGET_PART1)
	@$(RM) $(TARGET_PART2)

package:
	@echo "Packaging up project for submission..."
	@mkdir -p cse5441_lab4
	@cp $(SRC_PART1_DIR)/*.cu cse5441_lab4
	@cp $(SRC_PART2_DIR)/*.c cse5441_lab4
	@cp $(SRC_PART2_DIR)/*.h cse5441_lab4
	@cp $(SRC_PART2_DIR)/*.cu cse5441_lab4
	@cp submit.mk cse5441_lab4
	@mv cse5441_lab4/submit.mk cse5441_lab4/Makefile

	-include $(DEPS)
