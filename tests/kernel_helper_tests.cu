#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <kernel_helper/kernel_helper.cuh>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

bool gpu_available() {
  int count = 0;
  const cudaError_t result = cudaGetDeviceCount(&count);
  if (result != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return count > 0;
}

#define REQUIRE_GPU()                \
  do {                               \
    if (!gpu_available()) {          \
      GTEST_SKIP() << "No CUDA GPU"; \
    }                                \
  } while (false)

__global__ void index_kernel(std::uint64_t count, std::uint64_t* output) {
  kernel_helper::grid_stride_loop(count, [=] __device__(std::uint64_t index) {
    output[index] = kernel_helper::global_thread_id_1d();
  });
}

__global__ void vector_kernel(float3* output) {
  if (threadIdx.x == 0) {
    output[0] = kernel_helper::float3_ops::normalize_or_zero(make_float3(0, 0, 0));
    output[1] = kernel_helper::float3_ops::normalize_or_zero(make_float3(3, 0, 4));
    output[2] = kernel_helper::float3_ops::add(output[0], output[1]);
  }
}

__global__ void atomic_kernel(float* minimum, float* maximum) {
  const float value = static_cast<float>(threadIdx.x) - 17.0f;
  kernel_helper::atomic_min(minimum, value);
  kernel_helper::atomic_max(maximum, value);
  kernel_helper::atomic_min(minimum, nanf(""));
  kernel_helper::atomic_max(maximum, nanf(""));
}

template <int BlockThreads>
__global__ void block_collective_kernel(int valid_items, int* sum, int* minimum, int* maximum,
                                        int* generic) {
  __shared__ kernel_helper::block_reduce_temp_storage<int, BlockThreads> storage;
  const int value = static_cast<int>(threadIdx.x) + 1;

  const int sum_value = kernel_helper::block_sum<int, BlockThreads>(value, storage, valid_items);
  __syncthreads();
  if (threadIdx.x == 0) {
    *sum = sum_value;
  }

  const int min_value = kernel_helper::block_min<int, BlockThreads>(value, storage, valid_items);
  __syncthreads();
  if (threadIdx.x == 0) {
    *minimum = min_value;
  }

  const int max_value = kernel_helper::block_max<int, BlockThreads>(value, storage, valid_items);
  __syncthreads();
  if (threadIdx.x == 0) {
    *maximum = max_value;
  }

  const int generic_value = kernel_helper::block_reduce<int, BlockThreads>(
      value, storage, kernel_helper::maximum_op<int>{}, valid_items);
  if (threadIdx.x == 0) {
    *generic = generic_value;
  }
}

__global__ void warp_collective_kernel(int valid_items, int* sum, int* minimum, int* maximum,
                                       int* generic) {
  __shared__ kernel_helper::warp_reduce_temp_storage<int> storage;
  const int value = static_cast<int>(threadIdx.x) + 1;

  const int sum_value = kernel_helper::warp_sum(value, storage, valid_items);
  __syncwarp();
  if (kernel_helper::lane_id() == 0) {
    *sum = sum_value;
  }

  const int min_value = kernel_helper::warp_min(value, storage, valid_items);
  __syncwarp();
  if (kernel_helper::lane_id() == 0) {
    *minimum = min_value;
  }

  const int max_value = kernel_helper::warp_max(value, storage, valid_items);
  __syncwarp();
  if (kernel_helper::lane_id() == 0) {
    *maximum = max_value;
  }

  const int generic_value =
      kernel_helper::warp_reduce(value, storage, kernel_helper::maximum_op<int>{}, valid_items);
  if (kernel_helper::lane_id() == 0) {
    *generic = generic_value;
  }
}

TEST(Error, PreservesCallSiteContext) {
  const kernel_helper::cuda_error error(cudaErrorInvalidValue, "failing_call()", "sample.cu", 42);
  EXPECT_EQ(error.code(), cudaErrorInvalidValue);
  EXPECT_EQ(error.expression(), "failing_call()");
  EXPECT_EQ(error.file(), "sample.cu");
  EXPECT_EQ(error.line(), 42);
  EXPECT_NE(std::string(error.what()).find("sample.cu:42"), std::string::npos);
}

TEST(Math, ArithmeticUtilitiesHandleEdges) {
  EXPECT_EQ(kernel_helper::ceil_div<std::uint64_t>(0, 7), 0);
  EXPECT_EQ(kernel_helper::ceil_div<std::uint64_t>(15, 4), 4);
  EXPECT_EQ(kernel_helper::clamp(12, 0, 10), 10);
  EXPECT_FLOAT_EQ(kernel_helper::lerp(2.0f, 6.0f, 0.25f), 3.0f);
  EXPECT_FLOAT_EQ(kernel_helper::smoothstep(0.0f, 1.0f, 0.5f), 0.5f);
  EXPECT_TRUE(kernel_helper::approximately_equal(10.0, 10.01, 0.001, 0.002));
  EXPECT_TRUE(kernel_helper::is_power_of_two(1024u));
  EXPECT_FALSE(kernel_helper::is_power_of_two(1023u));
  EXPECT_EQ(kernel_helper::next_power_of_two(1025u), 2048u);
}

TEST(Memory, DetectsByteOverflowAndBoundsErrorsWithoutGpu) {
  EXPECT_THROW(
      kernel_helper::checked_byte_size<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
      std::length_error);
  EXPECT_NO_THROW(kernel_helper::check_range(10, 10, 0));
  EXPECT_THROW(kernel_helper::check_range(10, 8, 3), std::out_of_range);
  EXPECT_THROW(kernel_helper::check_range(10, 11, 0), std::out_of_range);
}

TEST(Launch, ZeroElementsProduceEmptyConfigurationWithoutGpuAccess) {
  const auto config = kernel_helper::make_launch_config_1d(0);
  EXPECT_TRUE(config.empty());
  EXPECT_FALSE(static_cast<bool>(config));
}

TEST(LaunchGpu, ValidatesNonEmptyConfigurationAgainstCurrentDevice) {
  REQUIRE_GPU();
  const auto config = kernel_helper::make_launch_config_1d(1003, 96);
  EXPECT_FALSE(config.empty());
  EXPECT_EQ(config.block.x, 96u);
  EXPECT_EQ(config.grid.x, 11u);
  EXPECT_THROW(kernel_helper::make_launch_config_1d(1, 0), std::invalid_argument);
}

TEST(MemoryGpu, OwnsMovesCopiesAndZeroFill) {
  REQUIRE_GPU();
  std::vector<int> input(257);
  std::iota(input.begin(), input.end(), 1);

  kernel_helper::device_buffer<int> first(input.size());
  kernel_helper::copy_h2d_async(first, input.data(), input.size());
  KERNEL_HELPER_CUDA_SYNCHRONIZE();

  kernel_helper::device_buffer<int> moved(std::move(first));
  EXPECT_TRUE(first.empty());
  EXPECT_EQ(moved.size(), input.size());

  kernel_helper::device_buffer<int> second(input.size());
  std::vector<int> output(input.size());
  kernel_helper::copy_d2d_async(second, moved, input.size());
  kernel_helper::copy_d2h_async(output.data(), second, output.size());
  KERNEL_HELPER_CUDA_SYNCHRONIZE();
  EXPECT_EQ(output, input);

  kernel_helper::zero_fill_async(second, second.size());
  KERNEL_HELPER_CUDA_SYNCHRONIZE();
  kernel_helper::copy_d2h(output.data(), second, output.size());
  EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](int value) { return value == 0; }));
  EXPECT_THROW(kernel_helper::copy_h2d(second, input.data(), 258), std::out_of_range);

  second.reset(8);
  EXPECT_EQ(second.size(), 8u);
  int* released = second.release();
  EXPECT_TRUE(second.empty());
  KERNEL_HELPER_CUDA_CHECK(cudaFree(released));
}

TEST(IndexingGpu, GridStrideLoopCoversNonPowerOfTwoInput) {
  REQUIRE_GPU();
  constexpr std::uint64_t count = 1003;
  kernel_helper::device_buffer<std::uint64_t> output(count);
  const auto config = kernel_helper::make_launch_config_1d(count, 96);
  index_kernel<<<config.grid, config.block>>>(count, output.data());
  KERNEL_HELPER_CUDA_CHECK_LAUNCH();
  KERNEL_HELPER_CUDA_SYNCHRONIZE();

  std::vector<std::uint64_t> host(count);
  kernel_helper::copy_d2h(host.data(), output, count);
  for (std::uint64_t index = 0; index < count; ++index) {
    EXPECT_LT(host[index], config.grid.x * config.block.x);
    EXPECT_EQ(host[index] % (config.grid.x * config.block.x), index);
  }
}

TEST(VectorGpu, NormalizationIsZeroSafe) {
  REQUIRE_GPU();
  kernel_helper::device_buffer<float3> output(3);
  vector_kernel<<<1, 1>>>(output.data());
  KERNEL_HELPER_CUDA_CHECK_LAUNCH();
  KERNEL_HELPER_CUDA_SYNCHRONIZE();

  float3 host[3]{};
  kernel_helper::copy_d2h(host, output, 3);
  EXPECT_FLOAT_EQ(host[0].x, 0.0f);
  EXPECT_FLOAT_EQ(host[0].y, 0.0f);
  EXPECT_FLOAT_EQ(host[0].z, 0.0f);
  EXPECT_FLOAT_EQ(host[1].x, 0.6f);
  EXPECT_FLOAT_EQ(host[1].z, 0.8f);
  EXPECT_FLOAT_EQ(host[2].x, 0.6f);
}

TEST(AtomicsGpu, MinMaxHandleContentionAndIgnoreIncomingNan) {
  REQUIRE_GPU();
  const float initial_min = std::numeric_limits<float>::infinity();
  const float initial_max = -std::numeric_limits<float>::infinity();
  kernel_helper::device_buffer<float> minimum(1);
  kernel_helper::device_buffer<float> maximum(1);
  kernel_helper::copy_h2d(minimum, &initial_min, 1);
  kernel_helper::copy_h2d(maximum, &initial_max, 1);

  atomic_kernel<<<1, 64>>>(minimum.data(), maximum.data());
  KERNEL_HELPER_CUDA_CHECK_LAUNCH();
  KERNEL_HELPER_CUDA_SYNCHRONIZE();

  float result_min = 0;
  float result_max = 0;
  kernel_helper::copy_d2h(&result_min, minimum, 1);
  kernel_helper::copy_d2h(&result_max, maximum, 1);
  EXPECT_FLOAT_EQ(result_min, -17.0f);
  EXPECT_FLOAT_EQ(result_max, 46.0f);
}

template <int BlockThreads>
void test_block_collective(int valid_items) {
  kernel_helper::device_buffer<int> output(4);
  block_collective_kernel<BlockThreads><<<1, BlockThreads>>>(
      valid_items, output.data(), output.data() + 1, output.data() + 2, output.data() + 3);
  KERNEL_HELPER_CUDA_CHECK_LAUNCH();
  KERNEL_HELPER_CUDA_SYNCHRONIZE();

  int host[4]{};
  kernel_helper::copy_d2h(host, output, 4);
  EXPECT_EQ(host[0], valid_items * (valid_items + 1) / 2);
  EXPECT_EQ(host[1], 1);
  EXPECT_EQ(host[2], valid_items);
  EXPECT_EQ(host[3], valid_items);
}

TEST(CollectivesGpu, BlockReductionsCoverMultipleBlockSizesAndPartialInput) {
  REQUIRE_GPU();
  test_block_collective<32>(19);
  test_block_collective<96>(77);
  test_block_collective<256>(231);
}

TEST(CollectivesGpu, WarpReductionsCoverPartialWarp) {
  REQUIRE_GPU();
  constexpr int valid_items = 19;
  kernel_helper::device_buffer<int> output(4);
  warp_collective_kernel<<<1, 32>>>(valid_items, output.data(), output.data() + 1,
                                    output.data() + 2, output.data() + 3);
  KERNEL_HELPER_CUDA_CHECK_LAUNCH();
  KERNEL_HELPER_CUDA_SYNCHRONIZE();

  int host[4]{};
  kernel_helper::copy_d2h(host, output, 4);
  EXPECT_EQ(host[0], valid_items * (valid_items + 1) / 2);
  EXPECT_EQ(host[1], 1);
  EXPECT_EQ(host[2], valid_items);
  EXPECT_EQ(host[3], valid_items);
}

}  // namespace
