#include <stdio.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "util.h"

#define assertm(exp, msg) assert((void(msg), exp))

void test_layouts() {
    printf("================ TEST LAYOUTS ================\n");
    printf("original layout: \n");
    // print(Layout<Shape<_8,_1024>,Stride<_1024,_1>>{});
    print_layout_2d_compressed(Layout<Shape<_8,_1024>,Stride<_1024,_1>>{});
    printf("\n");

    printf("Layout_K_SW128_Atom_Bits: \n");
    // print(SM90::GMMA::Layout_K_SW128_Atom_Bits{});
    print_layout_2d_compressed(SM90::GMMA::Layout_K_SW128_Atom_Bits{});
    printf("\n");

    printf("Layout_K_SW128_Atom: \n");
    print_layout_2d_compressed(SM90::GMMA::Layout_K_SW128_Atom<cutlass::float_e4m3_t>{});
    printf("\n");

    printf("8 x 128 layout: \n");
    print_layout_2d_compressed(ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_128>,Stride<_128,_1>>>{});
    printf("\n");
    printf("16 x 256 layout: \n");
    print_layout_2d_compressed(ComposedLayout<Swizzle<4,4,4>, smem_ptr_flag, Layout<Shape<_16,_256>,Stride<_256,_1>>>{});
    printf("\n");
}

template<typename Element>
void test_mma_layouts() {
    printf("******** %d BYTES ********\n", sizeof(Element));
    using TileShape_MNK = Shape<_128, _192, _128>;
    const int kStages = 2;

    auto smem_layout_q_atom = cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K,
        Element,
        decltype(cute::get<0>(TileShape_MNK{})),
        decltype(cute::get<2>(TileShape_MNK{}))
    >();
    auto smem_layout_q = tile_to_shape(smem_layout_q_atom, select<0, 2>(TileShape_MNK{}));
    printf("smem_layout_q_atom: \n");
    print_layout_2d_compressed<true>(smem_layout_q_atom);
    printf("\n");
    printf("smem_layout_q: \n");
    print_layout_2d_compressed<true>(smem_layout_q);
    printf("\n");

    auto smem_layout_k_atom = cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K,
        Element,
        decltype(cute::get<1>(TileShape_MNK{})),
        decltype(cute::get<2>(TileShape_MNK{}))
    >();
    auto smem_layout_k = tile_to_shape(
        smem_layout_k_atom,
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{}));
    printf("smem_layout_k_atom: \n");
    print_layout_2d_compressed<true>(smem_layout_k_atom);
    printf("\n");
    printf("smem_layout_k: \n");
    print_layout_3d_compressed<true>(smem_layout_k);
    printf("\n");


    using SmemLayoutAtomVt = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            GMMA::Major::MN,
            Element,
            decltype(cute::get<2>(TileShape_MNK{})),
            decltype(cute::get<1>(TileShape_MNK{}))>()
    );
    using SmemLayoutVt = decltype(tile_to_shape(
        SmemLayoutAtomVt{},
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
        cute::Step<_2, _1, _3>{}));

    printf("smem_layout_vt_atom: \n");
    print_layout_2d_compressed<false>(SmemLayoutAtomVt{});
    printf("\n");
    printf("smem_layout_vt: \n");
    print_layout_3d_compressed<false>(SmemLayoutVt{});
    printf("\n");

    using SmemLayoutAtomVtMma = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            GMMA::Major::K,
            Element,
            decltype(cute::get<2>(TileShape_MNK{})),
            decltype(cute::get<1>(TileShape_MNK{}))>()
    );
    using SmemLayoutVtMma = decltype(tile_to_shape(
        SmemLayoutAtomVtMma{},
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
        cute::Step<_1, _2, _3>{}));

    printf("smem_layout_vt_mma_atom: \n");
    print_layout_2d_compressed<true>(SmemLayoutAtomVtMma{});
    printf("\n");
    printf("smem_layout_vt_mma: \n");
    print_layout_3d_compressed<true>(SmemLayoutVtMma{});
    printf("\n");
}

void test_sf_layouts() {
    printf("================ TEST SF LAYOUTS ================\n");

    using SfConfig = cutlass::detail::Sm100BlockScaledConfig<32>;
    auto print_layout = [](int M, int N, int K, int L) {
        printf("**** problem_shape(%d, %d, %d, %d) ****\n", M, N, K, L);
        auto problem_shape = make_shape(M, N, K, L);
        // SFA shape: ((32,4), ceil(M/128)), ((SFVecSize,4), ceil(K/4), L)
        auto layout_sfa = SfConfig::tile_atom_to_shape_SFA(problem_shape);
        // SFB shape: ((32,4), ceil(N/128)), ((SFVecSize,4), ceil(K/4), L)
        auto layout_sfb = SfConfig::tile_atom_to_shape_SFB(problem_shape);
        printf("SFA: \n");
        print(layout_sfa);
        printf("\n");
        // print_layout_3d_compressed(layout_sfa);
        std::vector<std::tuple<int, int, int>> mapping(
            (int(std::ceil(float(K) / 128)) * 4) * (int(std::ceil(float(M) / 128)) * 128) * L,
            std::make_tuple(-1, -1, -1)
        );
        printf("mapping.size: %d\n", mapping.size());
        for (int b = 0; b < L; ++b) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < K; j += 32) {
                    int idx = layout_sfa(i, j, b);
                    printf("idx: %d, i: %d, j: %d, b: %d\n", idx, i, j, b);
                    assert(idx < mapping.size());
                    mapping[idx] = std::make_tuple(i, j, b);
                }
            }
        }
        for (int i = 0; i < mapping.size(); ++i) {
            auto [m, k, l] = mapping[i];
            printf("%d -> (%d, %d, %d)\n", i, m, k, l);
            fflush(stdout);
        }
    };

    // print_layout(128, 128, 128, 1);
    // print_layout(128, 128, 256, 1);
    // print_layout(128, 128, 129, 1);
    // print_layout(256, 128, 128, 1);
    // print_layout(129, 128, 128, 1);
    // print_layout(256, 128, 256, 1);
    print_layout(129, 128, 129, 1);
}

int main() {
    // test_layouts();
    // test_mma_layouts<cutlass::bfloat16_t>();
    // test_mma_layouts<cutlass::float_e4m3_t>();
    test_sf_layouts();
}
