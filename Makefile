# Set CUDA installation path (modify if needed)
CUDA_HOME ?= /usr/local/cuda-12.6
NVCC := $(CUDA_HOME)/bin/nvcc


# Compiler flags
CUDA_ARCH := -gencode arch=compute_90a,code=sm_90a  # Adjust for your GPU
CUDA_FLAGS := -std=c++17 $(CUDA_ARCH) -lineinfo \
              -O3 \
              --use_fast_math \
              --resource-usage \
              -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
              -DCUTLASS_ENABLE_GDC_FOR_SM90 \
              -DNDEBUG
              # --ftemplate-backtrace-limit=0  \ # To debug template code
              # --keep \
              # --ptxas-options=--verbose --register-usage-level=5 --warn-on-local-memory-usage \  # printing out number of registers
              # -DCUTLASS_DEBUG_TRACE_LEVEL=0 \  # Can toggle for debugging

# Directories
CUTLASS_DIR := third_party/cutlass/include
INCLUDE_DIR := include
SRC_DIR := src
BUILD_DIR := build

# Include and library paths
INCLUDE_DIRS := -I$(CUDA_HOME)/include -I$(CUTLASS_DIR) -I$(INCLUDE_DIR)
LIB_DIRS := -L$(CUDA_HOME)/lib64
LIBS := -lcudart -lcublas


# Binaries
BINARIES := $(BUILD_DIR)/layout


# Default target
all: $(BINARIES)

# Compile CUDA files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/layout: $(BUILD_DIR)/layout.o
	$(NVCC) $(BUILD_DIR)/layout.o -o $@ $(LIB_DIRS) $(LIBS)


# Clean up
clean:
	rm -f $(BINARIES)
