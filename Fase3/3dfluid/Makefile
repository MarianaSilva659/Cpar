# Define o shell como bash
SHELL := /bin/bash

# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -ffast-math -std=c++11
CFLAGS_CUDA = -O3 --use_fast_math -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets

# Directories
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = .

# Object files
OBJS = $(BUILD_DIR)/main.o $(BUILD_DIR)/EventManager.o $(BUILD_DIR)/fluid_solver.o

# Target executable
TARGET = $(BIN_DIR)/fluid_sim

# Default target
all: $(TARGET)

# Rule to create the executable
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	module load gcc/7.2.0 && module load cuda/11.3.1
	$(NVCC) $(CFLAGS_CUDA) $(OBJS) -o $(TARGET)

# Rule to compile .cpp source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	module load gcc/7.2.0
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile .cu source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	module load gcc/7.2.0 && module load cuda/11.3.1
	$(NVCC) $(CFLAGS_CUDA) -c $< -o $@

# Clean target to remove build and binary files
clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR) $(TARGET)
	@echo "Done."

# Run target for CUDA version
run: $(TARGET)
	@echo "Running CUDA application..."
	module load gcc/7.2.0 && module load cuda/11.3.1
	sbatch ./run.sh

# Phony targets
.PHONY: all clean runcuda
