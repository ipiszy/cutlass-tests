/**
I observed a case that loop-unrolling causes huge register spills when the loop modifies a variable inside the loop.
The purpose of this test is to reproduce this behavior.

Reported by ptxas:
ptxas info    : Compiling entry function '_Z27loop_unrolling_complex_goodPfS_S_' for 'sm_100a'
ptxas info    : Function properties for _Z27loop_unrolling_complex_goodPfS_S_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers, used 0 barriers
ptxas info    : Compile time = 20.092 ms
ptxas info    : Compiling entry function '_Z26loop_unrolling_complex_badPfS_S_' for 'sm_100a'
ptxas info    : Function properties for _Z26loop_unrolling_complex_badPfS_S_
    1024 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers, used 0 barriers, 1024 bytes cumulative stack size
ptxas info    : Compile time = 36.344 ms

The "good" loop interleaves LDG and other computations. The "bad" loop tries to do all LDGs at the beginning and 
move data from registers to local memory.
**/

#include <iostream>

#include "cute/arch/simd_sm100.hpp"
#include <cute/tensor.hpp>
#include "cutlass/array.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"

using namespace cute;

__global__ void loop_unrolling_complex_bad(float* in0, float* in1, float* res) {
  auto arr_SF_P_float = cute::make_tensor<float>(Int<4>{});
  auto tTMEM_LOADrS = cute::make_tensor<float>(Int<128>{});
  auto res_array = cute::make_tensor<int>(Int<128>{});

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 4; ++i) {
    arr_SF_P_float[i] = in0[i];
  }
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 128; ++i) {
    tTMEM_LOADrS[i] = in1[i];
  }

  const int inner_n = 32;
  const int outer_n = size(tTMEM_LOADrS) / inner_n;

  float2 minus_row_max_scale_fp32x2 = make_float2(-0.34, -0.34);
  float2 scale_fp32x2 = make_float2(-0.45, -0.45);

  cutlass::NumericConverter<float, int> convert;

  int offset_i = 0;

  CUTLASS_PRAGMA_UNROLL
  for (int offset = 0, outer_i = 0; offset < size(tTMEM_LOADrS); offset += inner_n, outer_i += 1) {
    float2 modified_scale;
    cute::add(
      modified_scale, minus_row_max_scale_fp32x2, 
      make_float2(-arr_SF_P_float[outer_i], -arr_SF_P_float[outer_i])
    );
    CUTLASS_PRAGMA_UNROLL
    // for (; offset_i < offset + inner_n; offset_i += 2) {
    // for (int i = 0; i < inner_n; i += 2) {
    for (; offset_i < offset + inner_n; offset_i += 2) {
      float2 in = make_float2(
        tTMEM_LOADrS(offset_i + 0),
        tTMEM_LOADrS(offset_i + 1)
      );
      float2 out;
      cute::fma(out, scale_fp32x2, in, modified_scale);
      tTMEM_LOADrS(offset_i + 0) = out.x;
      tTMEM_LOADrS(offset_i + 1) = out.y;
  
      tTMEM_LOADrS(offset_i + 0) = ::exp2f(tTMEM_LOADrS(offset_i + 0));
      tTMEM_LOADrS(offset_i + 1) = ::exp2f(tTMEM_LOADrS(offset_i + 1));
  
      res_array[(offset_i)] = convert(tTMEM_LOADrS(offset_i));
      res_array[(offset_i+1)] = convert(tTMEM_LOADrS(offset_i+1));
      // offset_i += 2;
    }
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 128; ++i) {
    *res += res_array[i];
  }
}


__global__ void loop_unrolling_complex_good(float* in0, float* in1, float* res) {
  auto arr_SF_P_float = cute::make_tensor<float>(Int<4>{});
  auto tTMEM_LOADrS = cute::make_tensor<float>(Int<128>{});
  auto res_array = cute::make_tensor<int>(Int<128>{});

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 4; ++i) {
    arr_SF_P_float[i] = in0[i];
  }
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 128; ++i) {
    tTMEM_LOADrS[i] = in1[i];
  }

  const int inner_n = 32;
  const int outer_n = size(tTMEM_LOADrS) / inner_n;

  float2 minus_row_max_scale_fp32x2 = make_float2(-0.34, -0.34);
  float2 scale_fp32x2 = make_float2(-0.45, -0.45);

  cutlass::NumericConverter<float, int> convert;

  CUTLASS_PRAGMA_UNROLL
  for (int offset = 0, outer_i = 0; offset < size(tTMEM_LOADrS); offset += inner_n, outer_i += 1) {
    float2 modified_scale;
    cute::add(
      modified_scale, minus_row_max_scale_fp32x2, 
      make_float2(-arr_SF_P_float[outer_i], -arr_SF_P_float[outer_i])
    );
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < inner_n; i += 2) {
      float2 in = make_float2(
        tTMEM_LOADrS(offset+i + 0),
        tTMEM_LOADrS(offset+i + 1)
      );
      float2 out;
      cute::fma(out, scale_fp32x2, in, modified_scale);
      tTMEM_LOADrS(offset+i + 0) = out.x;
      tTMEM_LOADrS(offset+i + 1) = out.y;
  
      tTMEM_LOADrS(offset+i + 0) = ::exp2f(tTMEM_LOADrS(offset+i + 0));
      tTMEM_LOADrS(offset+i + 1) = ::exp2f(tTMEM_LOADrS(offset+i + 1));
  
      res_array[(offset+i)] = convert(tTMEM_LOADrS(offset+i));
      res_array[(offset+i+1)] = convert(tTMEM_LOADrS(offset+i+1));
    }
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 128; ++i) {
    *res += res_array[i];
  }
}


/**
__global__ void loop_unrolling_good(float* in0, float* in1, float* res)
{
  const int outer_n = 128;
  const int inner_n = 32;

  *res = 0;
  auto in0_t = cute::make_tensor<float>(Int<4>{});
  auto in1_t = cute::make_tensor<float>(Int<128>{});
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 4; ++i) {
    in0_t[i] = in0[i];
  }
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 128; ++i) {
    in1_t[i] = in1[i];
  }

  CUTLASS_PRAGMA_UNROLL
  for (int offset = 0, outer_i = 0; offset < outer_n; offset += inner_n, outer_i += 1) {
    float modified_scale = outer_i * in0_t[outer_i];

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < inner_n; i += 2) {
      *res += modified_scale * in1_t[i + offset];
    }
  }
}

__global__ void loop_unrolling_bad(float* in0, float* in1, float* res)
{
  const int outer_n = 128;
  const int inner_n = 32;

  *res = 0;
  int offset_i = 0;
  auto in0_t = cute::make_tensor<float>(Int<4>{});
  auto in1_t = cute::make_tensor<float>(Int<128>{});
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 4; ++i) {
    in0_t[i] = in0[i];
  }
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 128; ++i) {
    in1_t[i] = in1[i];
  }

  CUTLASS_PRAGMA_UNROLL
  for (int offset = 0, outer_i = 0; offset < outer_n; offset += inner_n, outer_i += 1) {
    float modified_scale = outer_i * in0_t[outer_i];

    CUTLASS_PRAGMA_UNROLL
    for (; offset_i < offset + inner_n; offset_i += 2) {
      *res += modified_scale * in1_t[offset_i];
    }
  }
}
**/


int main(int argc, char** argv)
{
  double mean = 0.5;
  double stddev = 2.0;
  uint64_t seed = 0x2019;

  cutlass::HostTensor<float, cutlass::layout::RowMajor> in_tensor0({4, 1});
  cutlass::HostTensor<float, cutlass::layout::RowMajor> in_tensor1({128, 1});
  // Initialize in device memory
  cutlass::reference::device::TensorFillRandomGaussian(
    in_tensor0.device_view(),
    seed,
    (float)(mean),
    (float)(stddev));
  cutlass::reference::device::TensorFillRandomGaussian(
    in_tensor1.device_view(),
    seed,
    (float)(mean),
    (float)(stddev));

  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor({1, 1});

  // // Test loop unrolling - low register usage case.
  // loop_unrolling_good<<<1, 1>>>(in_tensor0.device_data(), in_tensor1.device_data(), tensor.device_data());
  // tensor.sync_host();
  // std::cout << "good, res: " << tensor.host_data()[0] << std::endl;
  // CUTE_CHECK_LAST();

  // // Test loop unrolling - high register usage case.
  // loop_unrolling_bad<<<1, 1>>>(in_tensor0.device_data(), in_tensor1.device_data(), tensor.device_data());
  // tensor.sync_host();
  // std::cout << "bad, res: " << tensor.host_data()[0] << std::endl;
  // CUTE_CHECK_LAST();

  // Test loop unrolling - complex usage case.
  loop_unrolling_complex_bad<<<1, 1>>>(in_tensor0.device_data(), in_tensor1.device_data(), tensor.device_data());
  tensor.sync_host();
  std::cout << "complex bad, res: " << tensor.host_data()[0] << std::endl;
  CUTE_CHECK_LAST();

  loop_unrolling_complex_good<<<1, 1>>>(in_tensor0.device_data(), in_tensor1.device_data(), tensor.device_data());
  tensor.sync_host();
  std::cout << "complex good, res: " << tensor.host_data()[0] << std::endl;
  CUTE_CHECK_LAST();

  return 0;
}
