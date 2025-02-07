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
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

using namespace cute;

enum Latency {
    empty_clk,
    get_ids,
    wgmma,
    raw_load_qk_prefetch,
    raw_load_q,
    cp_async_q_scale_token,
    cp_async_k_scale_token,
    prefetch_load_q,
    apply_prefetch_qk,
    apply_prefetch_qk_again,
    sm_load_q,
    raw_load_k,
    sm_load_k,
    sm_load_k_contiguous,
    __DUMMY_RES__,
    num_counters
};

enum BarrierID {
    producer_wait,
    consumer_wait
};

template <class ElementA,
          class ElementB,
          class ScaleType,
          class SmemLayoutA,  // (M,K)
          class SmemLayoutB,  // (N,K)
          class SmemLayoutQScaleToken,
          class SmemLayoutKScaleToken,
          class SmemLayoutKScaleBlock>
struct SharedStorage
{
  array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
  array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;
  array_aligned<ScaleType, cosize_v<SmemLayoutQScaleToken>> smem_Q_scale_token;
  array_aligned<ScaleType, cosize_v<SmemLayoutKScaleToken>> smem_K_scale_token;
  array_aligned<ScaleType, cosize_v<SmemLayoutKScaleBlock>> smem_K_scale_block;

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
    const ScaleType* scale_q_token,
    const ScaleType* scale_q_token_prefetch,
    const ScaleType* scale_k_token,
    const ScaleType* scale_k_token_prefetch,
    const ScaleType* scale_k_block,
    int64_t* latencies
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

    if (warp_group_idx == 0) {
        cutlass::arch::NamedBarrier::sync(128 * 2, BarrierID::producer_wait);
        if (warp_idx == 0) {
            int64_t clk0_2 = clock64();
            int idx = threadIdx.x << 1;
            cutlass::arch::cp_async_zfill<8 /* bytes */>(
                smem.smem_Q_scale_token.data() + idx, scale_q_token + idx, true);
            cutlass::arch::cp_async_wait<0>();
            int64_t clk0_3 = clock64();
            if (lane_predicate) {
                latencies[Latency::cp_async_q_scale_token] = clk0_3 - clk0_2;
            }
        } else if (warp_idx == 1 || warp_idx == 2) {
            int64_t clk12_2 = clock64();
            int idx = (threadIdx.x - 32) << 2;
            cutlass::arch::cp_async_zfill<16 /* bytes */>(
                smem.smem_K_scale_token.data() + idx, scale_k_token + idx, true);
            cutlass::arch::cp_async_wait<0>();
            int64_t clk12_3 = clock64();
            if (lane_predicate) {
                latencies[Latency::cp_async_k_scale_token] = clk12_3 - clk12_2;
            }
        }
        cutlass::arch::NamedBarrier::arrive(128 * 2, BarrierID::consumer_wait);
    } else if (warp_group_idx == 1) {
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

        auto l0 = tSrS.layout();
        auto new_l0 = make_layout(make_layout(get<0, 1>(l0), get<1>(l0)), make_layout(get<0, 0>(l0), get<0, 2>(l0), get<2>(l0)));
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), new_l0);

        Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
        auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);
        Tensor tScS = thread_mma.partition_C(cS);
        auto l1 = tScS.layout();
        auto new_l1 = make_layout(make_layout(get<0, 1>(l1), get<1>(l1)), make_layout(get<0, 0>(l1), get<0, 2>(l1), get<2>(l1)));
        Tensor tScS_rowcol = make_tensor(tScS.data(), new_l1);

        int64_t clk11 = clock64();

        Tensor scalesQToken = make_tensor<ScaleType>(size<0>(new_l0));
        Tensor scalesKToken = make_tensor<ScaleType>(size<1>(new_l1));
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = get<0>(tScS_rowcol(m, _0{}));
            scalesQToken(m) = scale_q_token_prefetch[row_idx];
        }
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            int const col_idx = get<1>(tScS_rowcol(_0{}, n));
            scalesKToken(n) = scale_k_token_prefetch[col_idx];
        }

        int64_t clk12 = clock64();

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

        float res = 0;

        int64_t clk3_0 = clock64();
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = get<0>(tScS_rowcol(m, _0{}));
            #pragma unroll
            for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                int const col_idx = get<1>(tScS_rowcol(_0{}, n));
                res += scalesQToken(m) + scalesKToken(n);
            }
        }
        int64_t clk3_1 = clock64();

        int64_t clk3_2 = clock64();
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = get<0>(tScS_rowcol(m, _0{}));
            #pragma unroll
            for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                int const col_idx = get<1>(tScS_rowcol(_0{}, n));
                res += scalesQToken(m) - scalesKToken(n);
            }
        }
        int64_t clk3_3 = clock64();

        int64_t clk3 = clock64();
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = get<0>(tScS_rowcol(m, _0{}));
            float scale = scale_q_token[row_idx];
            res += scale;
        }
        int64_t clk4 = clock64();

        cutlass::arch::NamedBarrier::arrive(128 * 2, BarrierID::producer_wait);
        cutlass::arch::NamedBarrier::sync(128 * 2, BarrierID::consumer_wait);

        int64_t clk5 = clock64();
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = get<0>(tScS_rowcol(m, _0{}));
            float scale = scale_q_token[row_idx];
            res += scale;
        }
        int64_t clk6 = clock64();

        int64_t clk7 = clock64();
        auto thread_lane = threadIdx.x % 4;
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = get<0>(tScS_rowcol(m, _0{}));
            // No bank conflicts because each thread accesses different banks within a warp.
            // 46 cycles. 2 fp32 accesses per thread, 23 cycles per access.
            float scale = smem.smem_Q_scale_token.data()[row_idx];
            res += scale;
        }
        int64_t clk8 = clock64();

        int64_t clk9 = clock64();
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            int const col_idx = get<1>(tScS_rowcol(_0{}, n));
            float scale = scale_k_token[col_idx];
            res += scale;
        }
        int64_t clk10 = clock64();

        int64_t clk13 = clock64();
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); n++) {
            int const col_idx = get<1>(tScS_rowcol(_0{}, n));
            // Different threads in a warp access same addresses inside a bank at the same time.
            // This won't cause bank conflict. Instead, broadcast will be triggered.
            // shared memory load k scale clock cycles: 275.
            // 56 fp32 per thread, 5 cycles per access? Maybe pipelining is effective.
            float scale = smem.smem_K_scale_token.data()[col_idx];
            // float scale = smem.smem_K_scale_token.data()[threadIdx.x % 3];

            // Different threads in a warp access different addresses inside a bank at the same time.
            // This will cause bank conflict.
            // float scale = smem.smem_K_scale_token.data()[threadIdx.x % 32 / 8 * 32 + threadIdx.x % 3];

            // CUTE_LOG("cycle: %d, addr: %p, load k scale, col_idx: %d, bank: %d\n", n, (void*)(smem.smem_K_scale_token.data() + col_idx), col_idx, col_idx % 32);
            res += scale;
        }
        int64_t clk14 = clock64();

        int64_t clk15 = clock64();
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); n++) {
            // shared memory load k scale clock cycles: 269.
            // This is 6 cycles smaller than the naive version above.
            // It seems that read throughput is the same, however this version reduces the number of instructions
            // from 28 (ld.shared.v2.f32) to 14 (ld.shared.v4.f32).
            // So maybe instruction launch latency is reduced a bit?
            // 56 fp32 per thread, 4.8 cycles per access?
            float scale = smem.smem_K_scale_token.data()[thread_lane * (kBlockN / 4) + n];

            // CUTE_LOG("cycle: %d, addr: %p, load k scale, col_idx: %d, bank: %d\n", n, (void*)(smem.smem_K_scale_token.data() + col_idx), col_idx, col_idx % 32);
            res += scale;
        }
        int64_t clk16 = clock64();


        latencies[Latency::__DUMMY_RES__] = res + tSrS_rowcol(_0{}, _0{});

        if (warp_idx == 4 && lane_predicate) {
            // print("tSrQ\n"); print(tSrQ); print("\n");
            // print("tSrK\n"); print(tSrK); print("\n");
            // print("tSrS\n"); print(tSrS); print("\n");
            // print("tScS\n"); print(tScS); print("\n");
            // print("tSrS_rowcol\n"); print(tSrS_rowcol); print("\n");
            // print("tScS_rowcol\n"); print(tScS_rowcol); print("\n");

            latencies[Latency::empty_clk] = clk1 - start;
            latencies[Latency::get_ids] = clk2 - clk1;
            latencies[Latency::raw_load_qk_prefetch] = clk12 - clk11;
            latencies[Latency::wgmma] = clk3_0 - clk12;
            latencies[Latency::apply_prefetch_qk] = clk3_1 - clk3_0;
            latencies[Latency::apply_prefetch_qk_again] = clk3_3 - clk3_2;
            latencies[Latency::raw_load_q] = clk4 - clk3;
            latencies[Latency::prefetch_load_q] = clk6 - clk5;
            latencies[Latency::sm_load_q] = clk8 - clk7;
            latencies[Latency::raw_load_k] = clk10 - clk9;
            latencies[Latency::sm_load_k] = clk14 - clk13;
            latencies[Latency::sm_load_k_contiguous] = clk16 - clk15;
        }
    }

    int64_t end = clock64();
}

void test_latency(int64_t* latencies) {
    const int kBlockM = 64;
    const int kBlockN = 224;
    const int kBlockNRounded = 256;
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
    thrust::device_vector<ScaleType> scale_q_token_prefetch(kBlockM);
    thrust::device_vector<ScaleType> scale_k_token(kBlockNRounded);
    thrust::device_vector<ScaleType> scale_k_token_prefetch(kBlockN);
    thrust::device_vector<ScaleType> scale_k_block(1);
    using SmemLayoutQScaleToken = Layout<Shape<Int<kBlockM>>>;
    using SmemLayoutKScaleToken = Layout<Shape<Int<kBlockNRounded>>>;
    using SmemLayoutKScaleBlock = Layout<Shape<_1>>;

    using SharedStorageT = SharedStorage<
        Element, Element, ScaleType, SmemLayoutQ, SmemLayoutK,
        SmemLayoutQScaleToken, SmemLayoutKScaleToken, SmemLayoutKScaleBlock
    >;
    // Launch parameter setup
    int smem_size = int(sizeof(SharedStorageT));
    dim3 dimBlock(size(TiledMma{}) + 128);
    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(round_up(1, dimCluster.x),
                 round_up(1, dimCluster.y));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

    void const* kernel_ptr = reinterpret_cast<void const*>(
        &test_latency_device<
            TiledMma, SmemLayoutQ, SmemLayoutK,
            SharedStorageT, TileShape_MNK, ScaleType
        >);

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));

    // Kernel Launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, kernel_ptr,
        scale_q_token_prefetch.data().get(), scale_q_token.data().get(),
        scale_k_token_prefetch.data().get(), scale_k_token.data().get(),
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

  std::cout << std::endl;
  std::cout << "empty clock cycles: " << h_latencies[Latency::empty_clk] << std::endl;
  std::cout << "get ids clock cycles: " << h_latencies[Latency::get_ids] << std::endl;
  std::cout << "wgmma clock cycles: " << h_latencies[Latency::wgmma] << std::endl;
  std::cout << std::endl;

  std::cout << "raw load qk scale prefetch clock cycles: " << h_latencies[Latency::raw_load_qk_prefetch] << std::endl;
  std::cout << "apply qk scale prefetch clock cycles: " << h_latencies[Latency::apply_prefetch_qk] << std::endl;
  std::cout << "apply qk scale prefetch again clock cycles: " << h_latencies[Latency::apply_prefetch_qk_again] << std::endl;
  std::cout << std::endl;

  std::cout << "raw load q scale clock cycles: " << h_latencies[Latency::raw_load_q] << std::endl;
  std::cout << "cp.async load q scale clock cycles (with L1 / L2 cache): " << h_latencies[Latency::cp_async_q_scale_token] << std::endl;
  std::cout << "prefetch load q scale clock cycles (with L1 / L2 cache): " << h_latencies[Latency::prefetch_load_q] << std::endl;
  std::cout << "shared memory load q scale clock cycles: " << h_latencies[Latency::sm_load_q] << std::endl;
  std::cout << std::endl;

  std::cout << "raw load k scale clock cycles: " << h_latencies[Latency::raw_load_k] << std::endl;
  std::cout << "shared memory load k scale clock cycles: " << h_latencies[Latency::sm_load_k] << std::endl;
  std::cout << "shared memory contiguous load k scale clock cycles: " << h_latencies[Latency::sm_load_k_contiguous] << std::endl;
  std::cout << std::endl;

  return 0;
}
