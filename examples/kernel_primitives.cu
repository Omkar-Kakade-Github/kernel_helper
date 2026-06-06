#include <cstdint>
#include <kernel_helper/kernel_helper.cuh>
#include <vector>

#include "example_validation.hpp"

namespace {

struct warp_metadata {
  std::uint64_t global_id;
  std::uint64_t global_stride;
  unsigned mask;
  int lane;
  int warp;
  int leader_lane;
  int explicit_leader;
  int active_leader;
};

struct vector_results {
  float3 uniform;
  float3 sum;
  float3 difference;
  float3 scaled;
  float dot;
  float length_squared;
  float length;
  float3 normalized;
  float3 zero_normalized;
};

__global__ void primitives_kernel(std::uint64_t count, std::uint64_t* owners,
                                  std::uint64_t* strides, warp_metadata* metadata,
                                  vector_results* vectors, int* zero_mask_results) {
  const std::uint64_t global_id = kernel_helper::global_thread_id_1d();
  const std::uint64_t global_stride = kernel_helper::global_thread_stride_1d();

  kernel_helper::grid_stride_loop(count, [=] __device__(std::uint64_t index) {
    owners[index] = global_id;
    strides[index] = global_stride;
  });

  const unsigned mask = kernel_helper::active_mask();
  metadata[global_id] = {
      global_id,
      global_stride,
      mask,
      kernel_helper::lane_id(),
      kernel_helper::warp_id(),
      kernel_helper::warp_leader_lane(mask),
      kernel_helper::is_warp_leader(mask) ? 1 : 0,
      kernel_helper::is_warp_leader() ? 1 : 0,
  };

  if (global_id == 0) {
    zero_mask_results[0] = kernel_helper::warp_leader_lane(0);
    zero_mask_results[1] = kernel_helper::is_warp_leader(0) ? 1 : 0;

    const float3 uniform = kernel_helper::float3_ops::uniform(2.0f);
    const float3 right = make_float3(1.0f, 3.0f, 5.0f);
    const float3 sum = kernel_helper::float3_ops::add(uniform, right);
    const float3 difference = kernel_helper::float3_ops::subtract(sum, right);
    const float3 scaled = kernel_helper::float3_ops::scale(right, 2.0f);
    const float3 normalized =
        kernel_helper::float3_ops::normalize_or_zero(make_float3(3.0f, 0.0f, 4.0f));
    const float3 zero_normalized =
        kernel_helper::float3_ops::normalize_or_zero(make_float3(0.0f, 0.0f, 0.0f));

    *vectors = {
        uniform,
        sum,
        difference,
        scaled,
        kernel_helper::float3_ops::dot(uniform, right),
        kernel_helper::float3_ops::length_squared(right),
        kernel_helper::float3_ops::length(right),
        normalized,
        zero_normalized,
    };
  }
}

__global__ void no_op_kernel() {}

void validate_float3(float3 value, float x, float y, float z) {
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(value.x, x, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(value.y, y, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(value.z, z, 1.0e-6f);
}

void validate_kernel_primitives() {
  const auto empty_config = kernel_helper::make_launch_config_1d(0);
  KERNEL_HELPER_EXAMPLE_REQUIRE(empty_config.empty());
  KERNEL_HELPER_EXAMPLE_REQUIRE(!static_cast<bool>(empty_config));

  const auto default_config = kernel_helper::make_launch_config_1d(1000);
  KERNEL_HELPER_EXAMPLE_REQUIRE(default_config);
  KERNEL_HELPER_EXAMPLE_REQUIRE(default_config.block.x == kernel_helper::default_threads_per_block);

  constexpr std::uint64_t count = 1003;
  const auto config = kernel_helper::make_launch_config_1d(count, 64);
  KERNEL_HELPER_EXAMPLE_REQUIRE(config);
  KERNEL_HELPER_EXAMPLE_REQUIRE(config.block.x == 64);
  KERNEL_HELPER_EXAMPLE_REQUIRE(config.grid.x == 16);

  const std::size_t launched_threads = static_cast<std::size_t>(config.grid.x) * config.block.x;
  kernel_helper::device_buffer<std::uint64_t> owners(count);
  kernel_helper::device_buffer<std::uint64_t> strides(count);
  kernel_helper::device_buffer<warp_metadata> metadata(launched_threads);
  kernel_helper::device_buffer<vector_results> vectors(1);
  kernel_helper::device_buffer<int> zero_mask_results(2);

  primitives_kernel<<<config.grid, config.block>>>(count, owners.data(), strides.data(),
                                                   metadata.data(), vectors.data(),
                                                   zero_mask_results.data());
  kernel_helper::check_last_launch(__FILE__, __LINE__);
  kernel_helper::synchronize_and_check(__FILE__, __LINE__);

  no_op_kernel<<<1, 1>>>();
  KERNEL_HELPER_CUDA_CHECK_LAUNCH();
  KERNEL_HELPER_CUDA_SYNCHRONIZE();

  std::vector<std::uint64_t> host_owners(count);
  std::vector<std::uint64_t> host_strides(count);
  std::vector<warp_metadata> host_metadata(launched_threads);
  vector_results host_vectors{};
  int host_zero_mask_results[2]{};
  kernel_helper::copy_d2h(host_owners.data(), owners, count);
  kernel_helper::copy_d2h(host_strides.data(), strides, count);
  kernel_helper::copy_d2h(host_metadata.data(), metadata, launched_threads);
  kernel_helper::copy_d2h(&host_vectors, vectors, 1);
  kernel_helper::copy_d2h(host_zero_mask_results, zero_mask_results, 2);

  for (std::uint64_t index = 0; index < count; ++index) {
    KERNEL_HELPER_EXAMPLE_REQUIRE(host_owners[index] == index % launched_threads);
    KERNEL_HELPER_EXAMPLE_REQUIRE(host_strides[index] == launched_threads);
  }

  for (std::size_t thread = 0; thread < launched_threads; ++thread) {
    const auto& value = host_metadata[thread];
    KERNEL_HELPER_EXAMPLE_REQUIRE(value.global_id == thread);
    KERNEL_HELPER_EXAMPLE_REQUIRE(value.global_stride == launched_threads);
    KERNEL_HELPER_EXAMPLE_REQUIRE(value.mask == 0xffffffffu);
    KERNEL_HELPER_EXAMPLE_REQUIRE(value.lane == static_cast<int>(thread % 32));
    KERNEL_HELPER_EXAMPLE_REQUIRE(value.warp == static_cast<int>((thread % 64) / 32));
    KERNEL_HELPER_EXAMPLE_REQUIRE(value.leader_lane == 0);
    KERNEL_HELPER_EXAMPLE_REQUIRE(value.explicit_leader == (value.lane == 0 ? 1 : 0));
    KERNEL_HELPER_EXAMPLE_REQUIRE(value.active_leader == (value.lane == 0 ? 1 : 0));
  }

  KERNEL_HELPER_EXAMPLE_REQUIRE(host_zero_mask_results[0] == -1);
  KERNEL_HELPER_EXAMPLE_REQUIRE(host_zero_mask_results[1] == 0);

  validate_float3(host_vectors.uniform, 2.0f, 2.0f, 2.0f);
  validate_float3(host_vectors.sum, 3.0f, 5.0f, 7.0f);
  validate_float3(host_vectors.difference, 2.0f, 2.0f, 2.0f);
  validate_float3(host_vectors.scaled, 2.0f, 6.0f, 10.0f);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(host_vectors.dot, 18.0f, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(host_vectors.length_squared, 35.0f, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(host_vectors.length, 5.9160798f, 1.0e-5f);
  validate_float3(host_vectors.normalized, 0.6f, 0.0f, 0.8f);
  validate_float3(host_vectors.zero_normalized, 0.0f, 0.0f, 0.0f);
}

}  // namespace

int main() {
  return example_validation::run("kernel primitives", true, validate_kernel_primitives);
}
