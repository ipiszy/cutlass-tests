#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

using namespace cute;

enum Latency {
    empty_clk,
    get_ids,
    wgmma,
    raw_load_q,
    __DUMMY_RES__,
    num_counters
};

template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K)
          class SmemLayoutB>  // (N,K)
struct SharedStorage
{
  array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
  array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;

  uint64_t mma_barrier;
};

template <
    typename TiledMma, typename SmemLayoutQ, typename SmemLayoutK,
    typename SharedStorage, typename TileShape_MNK, typename ScaleType
>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value + 128)
void
test_latency_device(
    const ScaleType* scale_q_token, const ScaleType* scale_k_token,
    const ScaleType* scale_k_block, int64_t* latencies
) {
    int64_t start = clock64();

    int64_t clk1 = clock64();
    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const warp_group_idx = cutlass::canonical_warp_group_idx();
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    // Shared memory tensors
    extern __shared__ char shared_memory[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    if (warp_group_idx == 1) {
        int64_t clk2 = clock64();

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int MmaWarpGroups = size(TiledMma{}) / cutlass::NumThreadsPerWarpGroup;

        TiledMma tiled_mma;
        Layout warp_group_thread_layout = make_layout(
            make_shape(Int<MmaWarpGroups>{}),
            make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        auto wg_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

        Tensor sQ = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutK{});
        Tensor tSrQ = wg_mma.partition_fragment_A(sQ);
        Tensor tSrK = wg_mma.partition_fragment_B(sK);
        Tensor tSrS = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));

        // MMAs to cover 1 K_TILE
        warpgroup_fence_operand(tSrS);
        warpgroup_arrive();
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
            gemm(tiled_mma, tSrQ(_, _, k_block), tSrK(_, _, k_block), tSrS);     // (V,M) x (V,N) => (V,M,N)
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();

        // Wait for all MMAs in a K_TILE to complete
        warpgroup_wait<0>();

        int64_t clk3 = clock64();

        float res = 0;

        auto l0 = tSrS.layout();
        auto new_l0 = make_layout(make_layout(get<0, 1>(l0), get<1>(l0)), make_layout(get<0, 0>(l0), get<0, 2>(l0), get<2>(l0)));
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), new_l0);

        Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
        auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);
        Tensor tScS = thread_mma.partition_C(cS);
        auto l1 = tScS.layout();
        auto new_l1 = make_layout(make_layout(get<0, 1>(l1), get<1>(l1)), make_layout(get<0, 0>(l1), get<0, 2>(l1), get<2>(l1)));
        Tensor tScS_rowcol = make_tensor(tScS.data(), new_l1);
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = get<0>(tScS_rowcol(m, _0{}));
            // for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            //     int const col_idx = get<1>(tScS_rowcol(_0{}, n));
            // }
            float scale = scale_q_token[row_idx];
            res += scale;
        }

        int64_t clk4 = clock64();
        latencies[Latency::__DUMMY_RES__] = res + tSrS_rowcol(_0{}, _0{});

        if (warp_idx == 4 && lane_predicate) {
            // print("tSrQ\n"); print(tSrQ); print("\n");
            // print("tSrK\n"); print(tSrK); print("\n");
            // print("tSrS\n"); print(tSrS); print("\n");
            // print("tScS\n"); print(tScS); print("\n");
            latencies[Latency::empty_clk] = clk1 - start;
            latencies[Latency::get_ids] = clk2 - clk1;
            latencies[Latency::wgmma] = clk3 - clk2;
            latencies[Latency::raw_load_q] = clk4 - clk3;
        }
    }

    int64_t end = clock64();
}

void test_latency(int64_t* latencies) {
    const int kBlockM = 64;
    const int kBlockN = 224;
    const int kHeadDim = 128;

    using Element = cutlass::float_e4m3_t;
    using ElementAccum = float;
    using ScaleType = float;

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
    using TiledMma = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutQK{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}))));

    thrust::device_vector<ScaleType> scale_q_token(kBlockM);
    thrust::device_vector<ScaleType> scale_k_token(kBlockN);
    thrust::device_vector<ScaleType> scale_k_block(1);

    // Launch parameter setup
    int smem_size = int(sizeof(SharedStorage<Element, Element, SmemLayoutQ, SmemLayoutK>));
    dim3 dimBlock(size(TiledMma{}) + 128);
    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(round_up(1, dimCluster.x),
                 round_up(1, dimCluster.y));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

    void const* kernel_ptr = reinterpret_cast<void const*>(
        &test_latency_device<
            TiledMma, SmemLayoutQ, SmemLayoutK,
            SharedStorage<Element, Element, SmemLayoutQ, SmemLayoutK>,
            TileShape_MNK, ScaleType
        >);

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));

    // Kernel Launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, kernel_ptr,
        scale_q_token.data().get(), scale_k_token.data().get(),
        scale_k_block.data().get(), latencies
    );
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
}

int main(int argc, char** argv)
{
  thrust::host_vector<int64_t> h_latencies(Latency::num_counters);
  thrust::device_vector<int64_t> d_latencies = h_latencies;

  // Run once
  test_latency(d_latencies.data().get());
  CUTE_CHECK_LAST();

  h_latencies = d_latencies;

  std::cout << "empty clock cycles: " << h_latencies[Latency::empty_clk] << std::endl;
  std::cout << "get ids clock cycles: " << h_latencies[Latency::get_ids] << std::endl;
  std::cout << "wgmma clock cycles: " << h_latencies[Latency::wgmma] << std::endl;
  std::cout << "raw load q scale clock cycles: " << h_latencies[Latency::raw_load_q] << std::endl;

  return 0;
}
