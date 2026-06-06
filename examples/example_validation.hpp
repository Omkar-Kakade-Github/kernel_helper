#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace example_validation {

inline constexpr int skip_return_code = 77;

inline void require(bool condition, const char* expression, const char* file, int line) {
  if (condition) {
    return;
  }

  std::ostringstream message;
  message << "validation failed: " << expression << " at " << file << ':' << line;
  throw std::runtime_error(message.str());
}

template <typename Left, typename Right, typename Tolerance>
void require_near(Left left, Right right, Tolerance tolerance, const char* expression,
                  const char* file, int line) {
  if (std::abs(left - right) <= tolerance) {
    return;
  }

  std::ostringstream message;
  message << "validation failed: " << expression << " at " << file << ':' << line
          << " (left=" << left << ", right=" << right << ", tolerance=" << tolerance << ')';
  throw std::runtime_error(message.str());
}

inline bool gpu_available(std::string& reason) {
  int device_count = 0;
  const cudaError_t result = cudaGetDeviceCount(&device_count);
  if (result != cudaSuccess) {
    reason = cudaGetErrorString(result);
    (void)cudaGetLastError();
    return false;
  }
  if (device_count == 0) {
    reason = "no CUDA devices were reported";
    return false;
  }
  return true;
}

template <typename Function>
int run(const char* name, bool requires_gpu, Function&& function) {
  if (requires_gpu) {
    std::string reason;
    if (!gpu_available(reason)) {
      std::cout << "[SKIP] " << name << ": " << reason << '\n';
      return skip_return_code;
    }
  }

  try {
    std::forward<Function>(function)();
    std::cout << "[PASS] " << name << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "[FAIL] " << name << ": unknown exception\n";
    return 1;
  }
}

}  // namespace example_validation

#define KERNEL_HELPER_EXAMPLE_REQUIRE(expression) \
  ::example_validation::require(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

#define KERNEL_HELPER_EXAMPLE_REQUIRE_NEAR(left, right, tolerance)                                \
  ::example_validation::require_near((left), (right), (tolerance), #left " ~= " #right, __FILE__, \
                                     __LINE__)
