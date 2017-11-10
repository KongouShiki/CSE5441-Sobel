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

INC_DIRS := $(shell find $(SRC_DIR) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CFLAGS := -O3
CUDAFLAGS := -O
CPPFLAGS := $(INC_FLAGS) -MMD -MP

all: $(TARGET_EXECUTABLES)

$(TARGET_PART1): $(OBJS)
	$(CUDA) $(CUDAFLAGS) -o $(BUILD_DIR)/Part1/maxwell_griffin_$@.o $(SRC_PART1_DIR)/maxwell_griffin_$@.cu
	$(CC) -o $@ $(BUILD_DIR)/Part1/maxwell_griffin_$@.o $(OBJS)

$(TARGET_PART2): $(OBJS)
	$(CUDA) $(CUDAFLAGS) -o $(BUILD_DIR)/Part2/maxwell_griffin_$(TARGET_PART2).o $(SRC_PART2_DIR)/maxwell_griffin_$(TARGET_PART2).cu
	$(CC) -o $@ $(BUILD_DIR)/Part2/maxwell_griffin_$(TARGET_PART2).o bmpReader.o $(OBJS)

# c source
$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@


.PHONY: clean package test

clean:
	@echo Cleaning build files...
	@$(RM) -r $(BUILD_DIR)
	@$(RM) $(TARGET_PART1)
	@$(RM) $(TARGET_PART2)

package:
	@echo "Packaging up project for submission..."
	@mkdir -p cse5441_lab4
	@cp Source/*.c Source/*.h Source/*/*.c Source/*/*.h cse5441_lab4
	@cp submit.mk cse5441_lab4
	@mv cse5441_lab4/submit.mk cse5441_lab4/Makefile


test:
	@mkdir -p Test
	@cp Source/Part1/maxwell_griffin_lab4p1.cu Test/test.c
	gcc Test/test.c -o test.out -Wall -std=c99
