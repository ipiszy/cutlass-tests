# Set CUDA installation path (modify if needed)
CUDA_HOME ?= /usr/local/cuda-12.8
NVCC := $(CUDA_HOME)/bin/nvcc


# Compiler flags
# CUDA_ARCH := -gencode arch=compute_90a,code=sm_90a  # Adjust for your GPU
CUDA_ARCH := -gencode arch=compute_100a,code=sm_100a  # Adjust for your GPU
CUDA_FLAGS := -std=c++17 $(CUDA_ARCH) -lineinfo \
              --use_fast_math \
              --resource-usage \
              --expt-relaxed-constexpr \
              --ptxas-options=--verbose \
              --keep \
              -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
              -DCUTLASS_ENABLE_GDC_FOR_SM90 \
              -DNDEBUG \
              -O3
	      # --debug --device-debug \
              # --ftemplate-backtrace-limit=0  \ # To debug template code
              # --keep \
              # --ptxas-options=--verbose --register-usage-level=5 --warn-on-local-memory-usage \  # printing out number of registers
              # -DCUTLASS_DEBUG_TRACE_LEVEL=0 \  # Can toggle for debugging

# Directories
INCLUDE_DIR := include
SRC_DIR := src
BUILD_DIR := build

# Include and library paths
INCLUDE_DIRS := -I$(CUDA_HOME)/include -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include -I$(INCLUDE_DIR)
LIB_DIRS := -L$(CUDA_HOME)/lib64
LIBS := -lcudart -lcublas


# Binaries
BINARIES := $(BUILD_DIR)/layout $(BUILD_DIR)/mem_latency $(BUILD_DIR)/cpasync $(BUILD_DIR)/loop_unrolling


# Default target
all: $(BINARIES)

# Compile CUDA files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/layout: $(BUILD_DIR)/layout.o
	$(NVCC) $(BUILD_DIR)/layout.o -o $@ $(LIB_DIRS) $(LIBS)

$(BUILD_DIR)/mem_latency: $(BUILD_DIR)/mem_latency.o
	$(NVCC) $(BUILD_DIR)/mem_latency.o -o $@ $(LIB_DIRS) $(LIBS)

$(BUILD_DIR)/cpasync: $(BUILD_DIR)/cp_async.o
	$(NVCC) $(BUILD_DIR)/cp_async.o -o $@ $(LIB_DIRS) $(LIBS)

$(BUILD_DIR)/loop_unrolling: $(BUILD_DIR)/loop_unrolling.o
	$(NVCC) $(BUILD_DIR)/loop_unrolling.o -o $@ $(LIB_DIRS) $(LIBS)


# Clean up
clean:
	rm -f $(BUILD_DIR)/*
