#include <kernel_helper/kernel_helper.cuh>

#include "example_validation.hpp"

namespace {

struct reduction_results {
  int generic;
  int sum;
  int minimum;
  int maximum;
};

template <int LogicalWarpThreads>
__global__ void warp_collectives_kernel(int valid_items, reduction_results* output) {
  __shared__ kernel_helper::warp_reduce_temp_storage<int, LogicalWarpThreads> storage;
  const int value = static_cast<int>(threadIdx.x) + 1;

  const int generic = kernel_helper::warp_reduce<int, LogicalWarpThreads>(
      value, storage, kernel_helper::maximum_op<int>{}, valid_items);
  if (kernel_helper::lane_id() == 0) {
    output->generic = generic;
  }
  __syncwarp();

  const int sum = kernel_helper::warp_sum<int, LogicalWarpThreads>(value, storage, valid_items);
  if (kernel_helper::lane_id() == 0) {
    output->sum = sum;
  }
  __syncwarp();

  const int minimum = kernel_helper::warp_min<int, LogicalWarpThreads>(value, storage, valid_items);
  if (kernel_helper::lane_id() == 0) {
    output->minimum = minimum;
  }
  __syncwarp();

  const int maximum = kernel_helper::warp_max<int, LogicalWarpThreads>(value, storage, valid_items);
  if (kernel_helper::lane_id() == 0) {
    output->maximum = maximum;
  }
}

template <int BlockThreads>
__global__ void block_collectives_kernel(int valid_items, reduction_results* output) {
  __shared__ kernel_helper::block_reduce_temp_storage<int, BlockThreads> storage;
  const int value = static_cast<int>(threadIdx.x) + 1;

  const int generic = kernel_helper::block_reduce<int, BlockThreads>(
      value, storage, kernel_helper::maximum_op<int>{}, valid_items);
  if (threadIdx.x == 0) {
    output->generic = generic;
  }
  __syncthreads();

  const int sum = kernel_helper::block_sum<int, BlockThreads>(value, storage, valid_items);
  if (threadIdx.x == 0) {
    output->sum = sum;
  }
  __syncthreads();

  const int minimum = kernel_helper::block_min<int, BlockThreads>(value, storage, valid_items);
  if (threadIdx.x == 0) {
    output->minimum = minimum;
  }
  __syncthreads();

  const int maximum = kernel_helper::block_max<int, BlockThreads>(value, storage, valid_items);
  if (threadIdx.x == 0) {
    output->maximum = maximum;
  }
}

void validate_results(const reduction_results& results, int valid_items) {
  KERNEL_HELPER_EXAMPLE_REQUIRE(results.generic == valid_items);
  KERNEL_HELPER_EXAMPLE_REQUIRE(results.sum == valid_items * (valid_items + 1) / 2);
  KERNEL_HELPER_EXAMPLE_REQUIRE(results.minimum == 1);
  KERNEL_HELPER_EXAMPLE_REQUIRE(results.maximum == valid_items);
}

template <int LogicalWarpThreads>
void run_warp_case(int valid_items) {
  kernel_helper::device_buffer<reduction_results> output(1);
  warp_collectives_kernel<LogicalWarpThreads>
      <<<1, LogicalWarpThreads>>>(valid_items, output.data());
  KERNEL_HELPER_CUDA_CHECK_LAUNCH();
  KERNEL_HELPER_CUDA_SYNCHRONIZE();

  reduction_results host{};
  kernel_helper::copy_d2h(&host, output, 1);
  validate_results(host, valid_items);
}

template <int BlockThreads>
void run_block_case(int valid_items) {
  kernel_helper::device_buffer<reduction_results> output(1);
  block_collectives_kernel<BlockThreads><<<1, BlockThreads>>>(valid_items, output.data());
  KERNEL_HELPER_CUDA_CHECK_LAUNCH();
  KERNEL_HELPER_CUDA_SYNCHRONIZE();

  reduction_results host{};
  kernel_helper::copy_d2h(&host, output, 1);
  validate_results(host, valid_items);
}

void validate_collectives() {
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::minimum_op<int>{}(7, 3) == 3);
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::maximum_op<int>{}(7, 3) == 7);

  run_warp_case<32>(32);
  run_warp_case<32>(19);
  run_block_case<32>(32);
  run_block_case<96>(77);
  run_block_case<256>(231);
}

}  // namespace

int main() {
  return example_validation::run("warp and block collectives", true, validate_collectives);
}
