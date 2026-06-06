#include <cstddef>
#include <cstdint>
#include <kernel_helper/kernel_helper.cuh>
#include <limits>
#include <stdexcept>
#include <string>

#include "example_validation.hpp"

namespace {

void validate_error_helpers() {
  kernel_helper::check_cuda(cudaSuccess, "cudaSuccess", __FILE__, __LINE__);

  bool direct_check_threw = false;
  try {
    kernel_helper::check_cuda(cudaErrorInvalidValue, "direct_failure()", "example.cu", 42);
  } catch (const kernel_helper::cuda_error& error) {
    direct_check_threw = true;
    KERNEL_HELPER_EXAMPLE_REQUIRE(error.code() == cudaErrorInvalidValue);
    KERNEL_HELPER_EXAMPLE_REQUIRE(error.expression() == "direct_failure()");
    KERNEL_HELPER_EXAMPLE_REQUIRE(error.file() == "example.cu");
    KERNEL_HELPER_EXAMPLE_REQUIRE(error.line() == 42);
    KERNEL_HELPER_EXAMPLE_REQUIRE(std::string(error.what()).find("example.cu:42") !=
                                  std::string::npos);
  }
  KERNEL_HELPER_EXAMPLE_REQUIRE(direct_check_threw);

  bool macro_check_threw = false;
  try {
    KERNEL_HELPER_CUDA_CHECK(cudaErrorInvalidValue);
  } catch (const kernel_helper::cuda_error& error) {
    macro_check_threw = true;
    KERNEL_HELPER_EXAMPLE_REQUIRE(error.code() == cudaErrorInvalidValue);
    KERNEL_HELPER_EXAMPLE_REQUIRE(error.expression() == "cudaErrorInvalidValue");
  }
  KERNEL_HELPER_EXAMPLE_REQUIRE(macro_check_threw);
}

void validate_memory_guards() {
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::checked_byte_size<std::uint32_t>(4) == 16);
  kernel_helper::check_range(10, 3, 7);
  kernel_helper::check_range(10, 10, 0);

  bool overflow_threw = false;
  try {
    (void)kernel_helper::checked_byte_size<std::uint64_t>(std::numeric_limits<std::size_t>::max());
  } catch (const std::length_error&) {
    overflow_threw = true;
  }
  KERNEL_HELPER_EXAMPLE_REQUIRE(overflow_threw);

  bool range_threw = false;
  try {
    kernel_helper::check_range(10, 8, 3);
  } catch (const std::out_of_range&) {
    range_threw = true;
  }
  KERNEL_HELPER_EXAMPLE_REQUIRE(range_threw);
}

void validate_math_helpers() {
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::ceil_div<std::uint64_t>(0, 7) == 0);
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::ceil_div<std::uint64_t>(15, 4) == 4);
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::minimum(7, 3) == 3);
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::maximum(7, 3) == 7);
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::clamp(12, 0, 10) == 10);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(kernel_helper::lerp(2.0f, 6.0f, 0.25f), 3.0f, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(kernel_helper::smoothstep(0.0f, 1.0f, 0.5f), 0.5f, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::absolute(-12) == 12);
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::approximately_equal(1.0, 1.0005, 0.001));
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::approximately_equal(1000.0, 1001.0, 0.1, 0.002));
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::is_power_of_two(1024u));
  KERNEL_HELPER_EXAMPLE_REQUIRE(!kernel_helper::is_power_of_two(1023u));
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::next_power_of_two(0u) == 1u);
  KERNEL_HELPER_EXAMPLE_REQUIRE(kernel_helper::next_power_of_two(1025u) == 2048u);
}

}  // namespace

int main() {
  return example_validation::run("error and math helpers", false, [] {
    validate_error_helpers();
    validate_memory_guards();
    validate_math_helpers();
  });
}
