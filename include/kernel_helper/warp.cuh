#pragma once

#include <cuda_runtime.h>

namespace kernel_helper {

__device__ __forceinline__ unsigned active_mask() noexcept { return __activemask(); }

__device__ __forceinline__ int lane_id() noexcept {
  return static_cast<int>(threadIdx.x) & (warpSize - 1);
}

__device__ __forceinline__ int warp_id() noexcept {
  return static_cast<int>(threadIdx.x) / warpSize;
}

__device__ __forceinline__ int warp_leader_lane(unsigned mask) noexcept {
  return __ffs(static_cast<int>(mask)) - 1;
}

__device__ __forceinline__ bool is_warp_leader(unsigned mask) noexcept {
  return mask != 0 && lane_id() == warp_leader_lane(mask);
}

__device__ __forceinline__ bool is_warp_leader() noexcept { return is_warp_leader(active_mask()); }

}  // namespace kernel_helper
