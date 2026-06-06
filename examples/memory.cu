#include <algorithm>
#include <cstddef>
#include <kernel_helper/kernel_helper.cuh>
#include <numeric>
#include <utility>
#include <vector>

#include "example_validation.hpp"

namespace {

void validate_memory_helpers() {
  constexpr std::size_t count = 64;
  std::vector<int> input(count);
  std::iota(input.begin(), input.end(), 1);
  std::vector<int> output(count, -1);

  kernel_helper::device_buffer<int> empty;
  KERNEL_HELPER_EXAMPLE_REQUIRE(empty.data() == nullptr);
  KERNEL_HELPER_EXAMPLE_REQUIRE(empty.size() == 0);
  KERNEL_HELPER_EXAMPLE_REQUIRE(empty.size_bytes() == 0);
  KERNEL_HELPER_EXAMPLE_REQUIRE(empty.empty());
  KERNEL_HELPER_EXAMPLE_REQUIRE(!static_cast<bool>(empty));

  kernel_helper::device_buffer<int> first(count);
  const kernel_helper::device_buffer<int>& const_first = first;
  KERNEL_HELPER_EXAMPLE_REQUIRE(first.data() != nullptr);
  KERNEL_HELPER_EXAMPLE_REQUIRE(const_first.data() == first.data());
  KERNEL_HELPER_EXAMPLE_REQUIRE(first.size() == count);
  KERNEL_HELPER_EXAMPLE_REQUIRE(first.size_bytes() == count * sizeof(int));
  KERNEL_HELPER_EXAMPLE_REQUIRE(!first.empty());
  KERNEL_HELPER_EXAMPLE_REQUIRE(static_cast<bool>(first));

  kernel_helper::copy_h2d(first.data(), input.data(), count);
  kernel_helper::copy_d2h(output.data(), first.data(), count);
  KERNEL_HELPER_EXAMPLE_REQUIRE(output == input);

  kernel_helper::device_buffer<int> second(count);
  kernel_helper::copy_d2d(second.data(), first.data(), count);
  std::fill(output.begin(), output.end(), -1);
  kernel_helper::copy_d2h(output.data(), second, count);
  KERNEL_HELPER_EXAMPLE_REQUIRE(output == input);

  std::vector<int> offset_values{101, 102, 103, 104};
  kernel_helper::copy_h2d(first, offset_values.data(), offset_values.size(), 8);
  std::vector<int> offset_output(offset_values.size());
  kernel_helper::copy_d2h(offset_output.data(), first, offset_output.size(), 8);
  KERNEL_HELPER_EXAMPLE_REQUIRE(offset_output == offset_values);

  kernel_helper::copy_d2d(second, first, offset_values.size(), 12, 8);
  std::fill(offset_output.begin(), offset_output.end(), 0);
  kernel_helper::copy_d2h(offset_output.data(), second, offset_output.size(), 12);
  KERNEL_HELPER_EXAMPLE_REQUIRE(offset_output == offset_values);

  kernel_helper::zero_fill(second.data() + 12, offset_values.size());
  kernel_helper::copy_d2h(offset_output.data(), second, offset_output.size(), 12);
  KERNEL_HELPER_EXAMPLE_REQUIRE(std::all_of(offset_output.begin(), offset_output.end(),
                                            [](int value) { return value == 0; }));

  kernel_helper::zero_fill(first, offset_values.size(), 8);
  kernel_helper::copy_d2h(offset_output.data(), first, offset_output.size(), 8);
  KERNEL_HELPER_EXAMPLE_REQUIRE(std::all_of(offset_output.begin(), offset_output.end(),
                                            [](int value) { return value == 0; }));

  cudaStream_t stream = nullptr;
  KERNEL_HELPER_CUDA_CHECK(cudaStreamCreate(&stream));
  try {
    kernel_helper::copy_h2d_async(first.data(), input.data(), count, stream);
    kernel_helper::copy_d2d_async(second.data(), first.data(), count, stream);
    kernel_helper::copy_d2h_async(output.data(), second.data(), count, stream);
    kernel_helper::synchronize_and_check(stream, __FILE__, __LINE__);
    KERNEL_HELPER_EXAMPLE_REQUIRE(output == input);

    std::vector<int> async_values{201, 202, 203, 204};
    kernel_helper::copy_h2d_async(first, async_values.data(), async_values.size(), stream, 16);
    kernel_helper::copy_d2d_async(second, first, async_values.size(), stream, 20, 16);
    kernel_helper::copy_d2h_async(output.data(), second, async_values.size(), stream, 20);
    KERNEL_HELPER_CUDA_STREAM_SYNCHRONIZE(stream);
    KERNEL_HELPER_EXAMPLE_REQUIRE(
        std::equal(async_values.begin(), async_values.end(), output.begin()));

    kernel_helper::zero_fill_async(second.data() + 20, async_values.size(), stream);
    kernel_helper::zero_fill_async(first, async_values.size(), stream, 16);
    KERNEL_HELPER_CUDA_STREAM_SYNCHRONIZE(stream);

    kernel_helper::copy_d2h(offset_output.data(), second, offset_output.size(), 20);
    KERNEL_HELPER_EXAMPLE_REQUIRE(std::all_of(offset_output.begin(), offset_output.end(),
                                              [](int value) { return value == 0; }));
    kernel_helper::copy_d2h(offset_output.data(), first, offset_output.size(), 16);
    KERNEL_HELPER_EXAMPLE_REQUIRE(std::all_of(offset_output.begin(), offset_output.end(),
                                              [](int value) { return value == 0; }));
  } catch (...) {
    (void)cudaStreamDestroy(stream);
    throw;
  }
  KERNEL_HELPER_CUDA_CHECK(cudaStreamDestroy(stream));

  kernel_helper::device_buffer<int> moved(std::move(first));
  KERNEL_HELPER_EXAMPLE_REQUIRE(first.empty());
  KERNEL_HELPER_EXAMPLE_REQUIRE(moved.size() == count);

  kernel_helper::device_buffer<int> move_assigned;
  move_assigned = std::move(moved);
  KERNEL_HELPER_EXAMPLE_REQUIRE(moved.empty());
  KERNEL_HELPER_EXAMPLE_REQUIRE(move_assigned.size() == count);

  kernel_helper::device_buffer<int> small(3);
  move_assigned.swap(small);
  KERNEL_HELPER_EXAMPLE_REQUIRE(move_assigned.size() == 3);
  KERNEL_HELPER_EXAMPLE_REQUIRE(small.size() == count);
  kernel_helper::swap(move_assigned, small);
  KERNEL_HELPER_EXAMPLE_REQUIRE(move_assigned.size() == count);
  KERNEL_HELPER_EXAMPLE_REQUIRE(small.size() == 3);

  small.reset(8);
  KERNEL_HELPER_EXAMPLE_REQUIRE(small.size() == 8);
  small.reset();
  KERNEL_HELPER_EXAMPLE_REQUIRE(small.empty());

  int* released = move_assigned.release();
  KERNEL_HELPER_EXAMPLE_REQUIRE(released != nullptr);
  KERNEL_HELPER_EXAMPLE_REQUIRE(move_assigned.empty());
  KERNEL_HELPER_CUDA_CHECK(cudaFree(released));

  kernel_helper::copy_h2d(static_cast<int*>(nullptr), static_cast<const int*>(nullptr), 0);
  kernel_helper::copy_d2h(static_cast<int*>(nullptr), static_cast<const int*>(nullptr), 0);
  kernel_helper::copy_d2d(static_cast<int*>(nullptr), static_cast<const int*>(nullptr), 0);
  kernel_helper::zero_fill(static_cast<int*>(nullptr), 0);
  kernel_helper::copy_h2d_async(static_cast<int*>(nullptr), static_cast<const int*>(nullptr), 0);
  kernel_helper::copy_d2h_async(static_cast<int*>(nullptr), static_cast<const int*>(nullptr), 0);
  kernel_helper::copy_d2d_async(static_cast<int*>(nullptr), static_cast<const int*>(nullptr), 0);
  kernel_helper::zero_fill_async(static_cast<int*>(nullptr), 0);
}

}  // namespace

int main() { return example_validation::run("memory helpers", true, validate_memory_helpers); }
