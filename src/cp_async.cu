/*
 * Test cp.async + cute behaviors for boundary cases.
 * Also refer to https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu.
 */

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

using namespace cute;

template <class Element, class SmemLayout>
struct SharedStorage
{
  array_aligned<Element, cosize_v<SmemLayout>> smem_inputs;
};

template <
    typename Element, typename CopyElement, typename GmemCopyAtom,
    typename GmemCopyThreadLayout, typename GmemCopyValLayout,
    typename SmemLayout, typename SharedStorageT, int NumLoadThreads>
__global__ static void test_cpasync_device(int num_elements, const Element* array) {
    // Shared memory tensors
    extern __shared__ char shared_memory[];
    SharedStorageT& smem = *reinterpret_cast<SharedStorageT*>(shared_memory);

    // Initialization.
    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const warp_group_idx = cutlass::canonical_warp_group_idx();

    // Copy
    auto tiled_copy = make_tiled_copy(GmemCopyAtom{}, GmemCopyThreadLayout{}, GmemCopyValLayout{});

    ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
    Tensor mTensors = make_tensor(make_gmem_ptr(array), cute::make_shape(_1{}, num_elements), cute::make_stride(_1{}, _1{}));  // (M, N)
    Tensor gTensors = group_modes<0, 2>(local_tile(
        domain_offset(make_coord(0, 0), mTensors),
        make_shape(_1{}), make_coord(_)));  // (((ATOM), ATOM_NUM), N), (((_1), _1), num_elements).
    Tensor tgTensors = thr_copy.partition_S(gTensors);  // ((COPY_ATOM_M, COPY_ATOM_N), NUM_M_ATOM, NUM_N_ATOM), ((_1,_1), _1, _1)
    Tensor sTensors = make_tensor(make_smem_ptr(smem.smem_inputs.data()), SmemLayout{});  // (kStages, block_size)
    Tensor tsTensors = thr_copy.partition_D(sTensors); // ((COPY_ATOM_M, COPY_ATOM_N), NUM_M_ATOM, NUM_N_ATOM), ((_1, _1), kStages, _1)

    if (warp_idx == 0 && lane_predicate) {
    // if (threadIdx.x == 96) {
        printf("num_elements: %d\n", num_elements);
        printf("tiled_copy: \n"); print(tiled_copy); print("\n");
        printf("thr_copy: \n"); print(thr_copy); print("\n");
        print("mTensors: \n"); print(mTensors); print("\n");
        print("gTensors: \n"); print(gTensors); print("\n");
        print("sTensors: \n"); print(sTensors); print("\n");
        print("tgTensors before group_modes: \n"); print(thr_copy.partition_S(gTensors)); print("\n");
        print("tsTensors before group_modes: \n"); print(thr_copy.partition_D(sTensors)); print("\n");
    }
    if (warp_idx == 0 && lane_predicate) {
    // if (threadIdx.x == 96) {
        // Clear sTensors first.
        clear(sTensors);
        // printf("dst tensors: \n");
        // print_tensor(sTensors);
        // printf("\n");
    }

    __syncthreads();
    if (threadIdx.x < NumLoadThreads) {
        // Without this if, the remaining threads will still do the copy even though
        // NUM_LOAD_THREADS is set in GmemCopyThreadLayout.
        copy(tiled_copy, tgTensors(_, _0{}, _0{}), tsTensors(_, _0{}, _0{}));
    }
    // When ZFILL is used, the remaining threads set smem to zeros, which is also not as expected.
    // Make an out-of-bound copy first.
    // copy(tiled_copy, tgTensors(_, _0{}, _0{}), tsTensors(_, _0{}, _0{}));
    // Check smem results.
    // copy(tiled_copy.with(threadIdx.x < NumLoadThreads), tgTensors(_, _0{}, _0{}), tsTensors(_, _0{}, _0{}));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    if (warp_idx == 0 && lane_predicate) {
    // if (threadIdx.x == 96) {
        printf("src tensors: \n");
        print_tensor(mTensors);
        printf("\n");

        printf("dst tensors: \n");
        print_tensor(sTensors);
        printf("\n");
    }
}

template <
    typename Element, typename CopyElement, typename GmemCopyAtom,
    typename GmemCopyThreadLayout, typename GmemCopyValLayout,
    typename SmemLayout, int NumThreads, int NumLoadThreads>
void launch_test_cpasync(int num_elements) {
    double mean = 0.5;
    double stddev = 2.0;
    uint64_t seed = 0x2019;

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor({1, num_elements});

    // Initialize in device memory
    cutlass::reference::device::TensorFillRandomGaussian(
      tensor.device_view(),
      seed,
      (Element)(mean),
      (Element)(stddev));

    using SharedStorageT = SharedStorage<Element, SmemLayout>;

    // Launch parameter setup
    int smem_size = int(sizeof(SharedStorageT));
    dim3 dimBlock(NumThreads);
    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(1);
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

    void const* kernel_ptr = reinterpret_cast<void const*>(
        &test_cpasync_device<Element, CopyElement, GmemCopyAtom, GmemCopyThreadLayout, GmemCopyValLayout, SmemLayout, SharedStorageT, NumLoadThreads>);

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));

    // Kernel Launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, kernel_ptr, num_elements, tensor.device_data()
    );
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
}


template <typename Element, typename CopyElement, int NumThreads, int NumLoadThreads>
void test_cpasync() {
    const static int kStages = 2;
    const static int kScaleElemsPerLoad = sizeof(CopyElement) / sizeof(Element);
    const static int kBlockSize = NumLoadThreads * kScaleElemsPerLoad;

    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<CopyElement>, Element>;
    using GmemCopyThreadLayout = Layout<Shape<_1, Int<NumLoadThreads>>, Stride<Int<NumLoadThreads>, _1>>;
    using GmemCopyValLayout = Layout<Shape<_1, Int<kScaleElemsPerLoad>>, Stride<Int<kScaleElemsPerLoad>, _1>>;
    using SmemLayout = Layout<Shape<Int<kStages>, Int<kBlockSize>>, Stride<Int<kBlockSize>, _1>>;

    printf("Full block copy: \n");
    launch_test_cpasync<Element, CopyElement, GmemCopyAtom, GmemCopyThreadLayout, GmemCopyValLayout, SmemLayout, NumThreads, NumLoadThreads>(kBlockSize);

    printf("Partial block copy: \n");
    launch_test_cpasync<Element, CopyElement, GmemCopyAtom, GmemCopyThreadLayout, GmemCopyValLayout, SmemLayout, NumThreads, NumLoadThreads>(kBlockSize / 2);

    printf("Multiple blocks copy: \n");
    launch_test_cpasync<Element, CopyElement, GmemCopyAtom, GmemCopyThreadLayout, GmemCopyValLayout, SmemLayout, NumThreads, NumLoadThreads>(kBlockSize * 2);
}

int main(int argc, char** argv)
{
  // Test the scenario when input tensor shape is smaller or larger than the defined GMEM copy size.
  // Conclusion: when ZFILL is used, fields are initialized to zeros properly.
  printf(" ==== Element: float, CopyElement: float,  num_load_threads = num_threads = 128 ==== \n");
  test_cpasync<float, float, 128, 128>();

  // Test the scenario when SMEM block size is smaller than a full block size, i.e. only a limited number
  // of threads are needed for loading.
  // Conclusion: Even though the GmemCopyThreadLayout is defined with NumLoadThreads, copy() doesn't really
  // checks whether a thread is within the limit of NumLoadThreads. As a result, need to add an if condition
  // to check threadIdx.x < NumLoadThreads before the copy.
  printf(" ==== Element: float, CopyElement: float,  num_load_threads = 54, num_threads = 128 ==== \n");
  test_cpasync<float, float, 128, 54>();

  // Tests multiple elements per load.
  printf(" ==== Element: fp8, CopyElement: uint128_t, num_load_threads = 4, num_threads = 128 ==== \n");
  test_cpasync<cutlass::float_e4m3_t, uint128_t, 128, 4>();
  CUTE_CHECK_LAST();

  return 0;
}
