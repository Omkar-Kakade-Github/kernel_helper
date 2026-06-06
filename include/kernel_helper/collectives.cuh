#pragma once

#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

namespace kernel_helper {

template <typename T>
struct minimum_op {
  __host__ __device__ constexpr T operator()(const T& left, const T& right) const {
    return right < left ? right : left;
  }
};

template <typename T>
struct maximum_op {
  __host__ __device__ constexpr T operator()(const T& left, const T& right) const {
    return left < right ? right : left;
  }
};

template <typename T, int LogicalWarpThreads = 32>
using warp_reduce_temp_storage = typename cub::WarpReduce<T, LogicalWarpThreads>::TempStorage;

template <typename T, int LogicalWarpThreads = 32, typename ReductionOp>
__device__ __forceinline__ T warp_reduce(T value,
                                         warp_reduce_temp_storage<T, LogicalWarpThreads>& storage,
                                         ReductionOp operation,
                                         int valid_items = LogicalWarpThreads) {
  return cub::WarpReduce<T, LogicalWarpThreads>(storage).Reduce(value, operation, valid_items);
}

template <typename T, int LogicalWarpThreads = 32>
__device__ __forceinline__ T warp_sum(T value,
                                      warp_reduce_temp_storage<T, LogicalWarpThreads>& storage,
                                      int valid_items = LogicalWarpThreads) {
  return cub::WarpReduce<T, LogicalWarpThreads>(storage).Sum(value, valid_items);
}

template <typename T, int LogicalWarpThreads = 32>
__device__ __forceinline__ T warp_min(T value,
                                      warp_reduce_temp_storage<T, LogicalWarpThreads>& storage,
                                      int valid_items = LogicalWarpThreads) {
  return warp_reduce<T, LogicalWarpThreads>(value, storage, minimum_op<T>{}, valid_items);
}

template <typename T, int LogicalWarpThreads = 32>
__device__ __forceinline__ T warp_max(T value,
                                      warp_reduce_temp_storage<T, LogicalWarpThreads>& storage,
                                      int valid_items = LogicalWarpThreads) {
  return warp_reduce<T, LogicalWarpThreads>(value, storage, maximum_op<T>{}, valid_items);
}

template <typename T, int BlockThreads,
          cub::BlockReduceAlgorithm Algorithm = cub::BLOCK_REDUCE_WARP_REDUCTIONS>
using block_reduce_temp_storage =
    typename cub::BlockReduce<T, BlockThreads, Algorithm>::TempStorage;

template <typename T, int BlockThreads,
          cub::BlockReduceAlgorithm Algorithm = cub::BLOCK_REDUCE_WARP_REDUCTIONS,
          typename ReductionOp>
__device__ __forceinline__ T
block_reduce(T value, block_reduce_temp_storage<T, BlockThreads, Algorithm>& storage,
             ReductionOp operation, int valid_items = BlockThreads) {
  return cub::BlockReduce<T, BlockThreads, Algorithm>(storage).Reduce(value, operation,
                                                                      valid_items);
}

template <typename T, int BlockThreads,
          cub::BlockReduceAlgorithm Algorithm = cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__device__ __forceinline__ T
block_sum(T value, block_reduce_temp_storage<T, BlockThreads, Algorithm>& storage,
          int valid_items = BlockThreads) {
  return cub::BlockReduce<T, BlockThreads, Algorithm>(storage).Sum(value, valid_items);
}

template <typename T, int BlockThreads,
          cub::BlockReduceAlgorithm Algorithm = cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__device__ __forceinline__ T
block_min(T value, block_reduce_temp_storage<T, BlockThreads, Algorithm>& storage,
          int valid_items = BlockThreads) {
  return block_reduce<T, BlockThreads, Algorithm>(value, storage, minimum_op<T>{}, valid_items);
}

template <typename T, int BlockThreads,
          cub::BlockReduceAlgorithm Algorithm = cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__device__ __forceinline__ T
block_max(T value, block_reduce_temp_storage<T, BlockThreads, Algorithm>& storage,
          int valid_items = BlockThreads) {
  return block_reduce<T, BlockThreads, Algorithm>(value, storage, maximum_op<T>{}, valid_items);
}

}  // namespace kernel_helper
