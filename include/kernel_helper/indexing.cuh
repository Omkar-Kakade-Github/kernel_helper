#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

namespace kernel_helper {

__device__ __forceinline__ std::uint64_t global_thread_id_1d() noexcept {
  return static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ std::uint64_t global_thread_stride_1d() noexcept {
  return static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
}

template <typename Function>
__device__ __forceinline__ void grid_stride_loop(std::uint64_t count, Function&& function) {
  const std::uint64_t stride = global_thread_stride_1d();
  for (std::uint64_t index = global_thread_id_1d(); index < count; index += stride) {
    function(index);
  }
}

}  // namespace kernel_helper
