#pragma once

#include <cuda_runtime.h>

#include <type_traits>

namespace kernel_helper {

template <typename T>
__host__ __device__ constexpr T ceil_div(T numerator, T denominator) noexcept {
  static_assert(std::is_integral<T>::value, "ceil_div requires an integral type");
  return numerator == T{0} ? T{0} : static_cast<T>(T{1} + (numerator - T{1}) / denominator);
}

template <typename T>
__host__ __device__ constexpr T minimum(T a, T b) noexcept {
  return b < a ? b : a;
}

template <typename T>
__host__ __device__ constexpr T maximum(T a, T b) noexcept {
  return a < b ? b : a;
}

template <typename T>
__host__ __device__ constexpr T clamp(T value, T lower, T upper) noexcept {
  return minimum(maximum(value, lower), upper);
}

template <typename T, typename U>
__host__ __device__ constexpr T lerp(T a, T b, U amount) noexcept {
  return static_cast<T>(a + amount * (b - a));
}

template <typename T>
__host__ __device__ constexpr T smoothstep(T edge0, T edge1, T value) noexcept {
  const T t = clamp((value - edge0) / (edge1 - edge0), T{0}, T{1});
  return t * t * (T{3} - T{2} * t);
}

template <typename T>
__host__ __device__ constexpr T absolute(T value) noexcept {
  return value < T{0} ? -value : value;
}

template <typename T>
__host__ __device__ constexpr bool approximately_equal(T a, T b, T absolute_tolerance,
                                                       T relative_tolerance = T{0}) noexcept {
  const T difference = absolute(a - b);
  const T scale = maximum(absolute(a), absolute(b));
  return difference <= maximum(absolute_tolerance, relative_tolerance * scale);
}

template <typename T>
__host__ __device__ constexpr bool is_power_of_two(T value) noexcept {
  static_assert(std::is_integral<T>::value, "is_power_of_two requires an integral type");
  return value > T{0} && (value & (value - T{1})) == T{0};
}

template <typename T>
__host__ __device__ constexpr T next_power_of_two(T value) noexcept {
  static_assert(std::is_integral<T>::value, "next_power_of_two requires an integral type");
  static_assert(std::is_unsigned<T>::value, "next_power_of_two requires an unsigned type");

  if (value <= T{1}) {
    return T{1};
  }

  --value;
  for (unsigned shift = 1; shift < sizeof(T) * 8; shift <<= 1) {
    value |= value >> shift;
  }
  return ++value;
}

}  // namespace kernel_helper
