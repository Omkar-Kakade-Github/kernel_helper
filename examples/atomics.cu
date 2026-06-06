#include <cmath>
#include <kernel_helper/kernel_helper.cuh>
#include <limits>

#include "example_validation.hpp"

namespace {

struct atomic_values {
  float float_min;
  float float_max;
  double double_min;
  double double_max;
};

struct atomic_returns {
  float float_min_replace;
  float float_min_nan;
  float float_max_replace;
  float float_max_nan;
  double double_min_replace;
  double double_min_nan;
  double double_max_replace;
  double double_max_nan;
};

__global__ void sequential_atomic_kernel(atomic_values* values, atomic_returns* returns) {
  if (threadIdx.x != 0) {
    return;
  }

  returns->float_min_replace = kernel_helper::atomic_min(&values->float_min, 5.0f);
  returns->float_min_nan = kernel_helper::atomic_min(&values->float_min, nanf(""));
  returns->float_max_replace = kernel_helper::atomic_max(&values->float_max, 6.0f);
  returns->float_max_nan = kernel_helper::atomic_max(&values->float_max, nanf(""));

  returns->double_min_replace = kernel_helper::atomic_min(&values->double_min, 7.0);
  returns->double_min_nan = kernel_helper::atomic_min(&values->double_min, nan(""));
  returns->double_max_replace = kernel_helper::atomic_max(&values->double_max, 8.0);
  returns->double_max_nan = kernel_helper::atomic_max(&values->double_max, nan(""));
}

__global__ void contended_atomic_kernel(atomic_values* values) {
  const float float_value = static_cast<float>(threadIdx.x) - 17.0f;
  const double double_value = static_cast<double>(threadIdx.x) - 23.0;
  kernel_helper::atomic_min(&values->float_min, float_value);
  kernel_helper::atomic_max(&values->float_max, float_value);
  kernel_helper::atomic_min(&values->double_min, double_value);
  kernel_helper::atomic_max(&values->double_max, double_value);
}

void validate_atomics() {
  const atomic_values nan_values{
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
  };

  kernel_helper::device_buffer<atomic_values> values(1);
  kernel_helper::device_buffer<atomic_returns> returns(1);
  kernel_helper::copy_h2d(values, &nan_values, 1);

  sequential_atomic_kernel<<<1, 1>>>(values.data(), returns.data());
  KERNEL_HELPER_CUDA_CHECK_LAUNCH();
  KERNEL_HELPER_CUDA_SYNCHRONIZE();

  atomic_values sequential_values{};
  atomic_returns sequential_returns{};
  kernel_helper::copy_d2h(&sequential_values, values, 1);
  kernel_helper::copy_d2h(&sequential_returns, returns, 1);

  KERNEL_HELPER_EXAMPLE_REQUIRE(std::isnan(sequential_returns.float_min_replace));
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(sequential_returns.float_min_nan, 5.0f, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE(std::isnan(sequential_returns.float_max_replace));
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(sequential_returns.float_max_nan, 6.0f, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE(std::isnan(sequential_returns.double_min_replace));
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(sequential_returns.double_min_nan, 7.0, 1.0e-12);
  KERNEL_HELPER_EXAMPLE_REQUIRE(std::isnan(sequential_returns.double_max_replace));
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(sequential_returns.double_max_nan, 8.0, 1.0e-12);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(sequential_values.float_min, 5.0f, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(sequential_values.float_max, 6.0f, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(sequential_values.double_min, 7.0, 1.0e-12);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(sequential_values.double_max, 8.0, 1.0e-12);

  const atomic_values initial_values{
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
  };
  kernel_helper::copy_h2d(values, &initial_values, 1);
  contended_atomic_kernel<<<1, 64>>>(values.data());
  KERNEL_HELPER_CUDA_CHECK_LAUNCH();
  KERNEL_HELPER_CUDA_SYNCHRONIZE();

  atomic_values contended_values{};
  kernel_helper::copy_d2h(&contended_values, values, 1);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(contended_values.float_min, -17.0f, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(contended_values.float_max, 46.0f, 1.0e-6f);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(contended_values.double_min, -23.0, 1.0e-12);
  KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(contended_values.double_max, 40.0, 1.0e-12);
}

}  // namespace

int main() { return example_validation::run("floating-point atomics", true, validate_atomics); }
