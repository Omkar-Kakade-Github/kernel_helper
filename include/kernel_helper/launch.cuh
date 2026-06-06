#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <kernel_helper/error.cuh>
#include <kernel_helper/math.cuh>
#include <limits>
#include <stdexcept>

namespace kernel_helper {

struct launch_config_1d {
  dim3 grid{0, 1, 1};
  dim3 block{0, 1, 1};

  constexpr bool empty() const noexcept { return grid.x == 0 || block.x == 0; }
  constexpr explicit operator bool() const noexcept { return !empty(); }
};

inline launch_config_1d make_launch_config_1d(std::uint64_t element_count,
                                              unsigned threads_per_block = 256) {
  if (element_count == 0) {
    return {};
  }
  if (threads_per_block == 0) {
    throw std::invalid_argument("threads_per_block must be greater than zero");
  }

  int device = 0;
  KERNEL_HELPER_CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp properties{};
  KERNEL_HELPER_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  if (threads_per_block > static_cast<unsigned>(properties.maxThreadsPerBlock)) {
    throw std::invalid_argument("threads_per_block exceeds the current device limit");
  }

  const std::uint64_t required_blocks =
      ceil_div(element_count, static_cast<std::uint64_t>(threads_per_block));
  const std::uint64_t device_grid_limit = static_cast<std::uint64_t>(properties.maxGridSize[0]);
  const std::uint64_t dim3_limit = static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max());
  const auto block_count =
      static_cast<unsigned>(minimum(required_blocks, minimum(device_grid_limit, dim3_limit)));

  return {dim3(block_count), dim3(threads_per_block)};
}

}  // namespace kernel_helper
