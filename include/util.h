#include <cassert>
#include <string>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

using namespace cute;

namespace {

std::string label(int start_idx, int end_idx) {
    int step_size = end_idx - start_idx;
    std::string s = std::to_string(start_idx) + (
        step_size > 0 ?
        std::string("~") + std::to_string(end_idx) :
        std::string("")
    );
    return s;
}

}  // namespace

template <bool IsKMajor=true, class Layout>
void print_layout_2d_compressed(Layout const& layout) {
  CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  int idx_width = num_digits(cosize(layout)) * 2 + 4;
  const char* delim = "+-----------------------";

  print(layout); print("\n");

  const bool is_k_major = IsKMajor;
  const int size_m = is_k_major ? 0 : 1;
  const int size_n = is_k_major ? 1 : 0;

  // Compute max_contiguous_num_elements.
  int max_contiguous_num_elements = -1;
  for (int m = 0; m < size<size_m>(layout); ++m) {
    int current_contiguous_num_elements = 0;
    int last_value = -1;
    for (int n = 0; n < size<size_n>(layout); ++n) {
        int curr_value = is_k_major ? int(layout(m, n)) : int(layout(n, m));
        if (n == 0) {
            current_contiguous_num_elements = 1;
        } else if (curr_value != last_value + 1) {
            if (max_contiguous_num_elements < 0 || current_contiguous_num_elements < max_contiguous_num_elements) {
                max_contiguous_num_elements = current_contiguous_num_elements;
                current_contiguous_num_elements = 1;
            }
        } else {
            current_contiguous_num_elements += 1;
        }
        last_value = curr_value;
    }
    if (max_contiguous_num_elements < 0 || current_contiguous_num_elements < max_contiguous_num_elements) {
        max_contiguous_num_elements = current_contiguous_num_elements;
    }
  }

  // Print out indices
  const int n_step_size = is_k_major ? max_contiguous_num_elements : 1;
  const int m_step_size = is_k_major ? 1 : max_contiguous_num_elements;

  // Column indices
  print("%*s", idx_width, " ");
  for (int n = 0; n < size<1>(layout); n += n_step_size) {
    printf("  %*s ", idx_width-2, label(n, n + n_step_size - 1).c_str());
  }
  printf("\n");

  // Print out A m-by-n
  for (int m = 0; m < size<0>(layout); m += m_step_size) {
    // Header
    print("%*s", idx_width, " ");
    for (int n = 0; n < size<1>(layout); n += n_step_size) {
        printf("%.*s", idx_width+1, delim);
    }
    printf("+\n");
    // Values
    printf("%*s", idx_width, label(m, m + m_step_size - 1).c_str());  // Row indices
    for (int n = 0; n < size<1>(layout); n += n_step_size) {
        int start_idx = int(layout(m,n));
        int end_idx = start_idx + m_step_size - 1 + n_step_size - 1;
        assert(end_idx == int(layout(m + m_step_size - 1, n + n_step_size - 1)));
        printf("| %*s ", idx_width-2, label(start_idx, end_idx).c_str());
    }
    printf("|\n");
  }

  // Footer
  print("%*s", idx_width, " ");
  for (int n = 0; n < size<1>(layout); n += n_step_size) { printf("%.*s", idx_width+1, delim); }
  printf("+\n");


  // Print out chunk indicies
  if (rank(get<size_n>(layout)) != 1) {
    return;
  }

  print("\n");
  // Column indices
  print("%*s", idx_width, " ");
  for (int n = 0; n < size<1>(layout); n += n_step_size) {
    printf("  %*s ", idx_width-2, label(n, n + n_step_size - 1).c_str());
  }
  printf("\n");

  // Print out A m-by-n
  for (int m = 0; m < size<0>(layout); m += m_step_size) {
    // Header
    print("%*s", idx_width, " ");
    for (int n = 0; n < size<1>(layout); n += n_step_size) {
        printf("%.*s", idx_width+1, delim);
    }
    printf("+\n");
    // Values
    printf("%*s", idx_width, label(m, m + m_step_size - 1).c_str());  // Row indices
    for (int n = 0; n < size<1>(layout); n += n_step_size) {
        int base = (
          is_k_major ? 
          int(layout(m, n)) - m * size<1>(layout) : 
          int(layout(m, n)) - n * size<0>(layout)
        );
        int chunk_idx = base / max_contiguous_num_elements;
        printf("| %*d ", idx_width-2, chunk_idx);
    }
    printf("|\n");
  }

  // Footer
  print("%*s", idx_width, " ");
  for (int n = 0; n < size<1>(layout); n += n_step_size) { printf("%.*s", idx_width+1, delim); }
  printf("+\n");
}

template <bool IsKMajor=true, class Layout>
void print_layout_3d_compressed(Layout const& layout) {
    CUTE_STATIC_ASSERT_V(rank(layout) == Int<3>{});
    print(layout); print("\n");
    for (int m = 0; m < size<2>(layout); ++m) {
        print_layout_2d_compressed<IsKMajor>(layout(_, _, m));
    }
}
