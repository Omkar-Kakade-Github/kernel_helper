#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace kernel_helper {

// Incoming NaNs are ignored. A stored NaN is replaced by a finite value.
__device__ inline float atomic_min(float* address, float value) noexcept {
  auto* bits = reinterpret_cast<unsigned int*>(address);
  unsigned int old = atomicCAS(bits, 0u, 0u);
  if (isnan(value)) {
    return __uint_as_float(old);
  }

  while (true) {
    const float current = __uint_as_float(old);
    const float desired = isnan(current) || value < current ? value : current;
    const unsigned int assumed = old;
    old = atomicCAS(bits, assumed, __float_as_uint(desired));
    if (old == assumed) {
      return current;
    }
  }
}

__device__ inline float atomic_max(float* address, float value) noexcept {
  auto* bits = reinterpret_cast<unsigned int*>(address);
  unsigned int old = atomicCAS(bits, 0u, 0u);
  if (isnan(value)) {
    return __uint_as_float(old);
  }

  while (true) {
    const float current = __uint_as_float(old);
    const float desired = isnan(current) || current < value ? value : current;
    const unsigned int assumed = old;
    old = atomicCAS(bits, assumed, __float_as_uint(desired));
    if (old == assumed) {
      return current;
    }
  }
}

__device__ inline double atomic_min(double* address, double value) noexcept {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = atomicCAS(bits, 0ull, 0ull);
  if (isnan(value)) {
    return __longlong_as_double(static_cast<long long>(old));
  }

  while (true) {
    const double current = __longlong_as_double(static_cast<long long>(old));
    const double desired = isnan(current) || value < current ? value : current;
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed, static_cast<unsigned long long>(__double_as_longlong(desired)));
    if (old == assumed) {
      return current;
    }
  }
}

__device__ inline double atomic_max(double* address, double value) noexcept {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = atomicCAS(bits, 0ull, 0ull);
  if (isnan(value)) {
    return __longlong_as_double(static_cast<long long>(old));
  }

  while (true) {
    const double current = __longlong_as_double(static_cast<long long>(old));
    const double desired = isnan(current) || current < value ? value : current;
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed, static_cast<unsigned long long>(__double_as_longlong(desired)));
    if (old == assumed) {
      return current;
    }
  }
}

}  // namespace kernel_helper
