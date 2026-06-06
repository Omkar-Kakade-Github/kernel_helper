#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace kernel_helper {
namespace float3_ops {

__host__ __device__ inline float3 uniform(float value) noexcept {
  return make_float3(value, value, value);
}

__host__ __device__ inline float3 add(float3 left, float3 right) noexcept {
  return make_float3(left.x + right.x, left.y + right.y, left.z + right.z);
}

__host__ __device__ inline float3 subtract(float3 left, float3 right) noexcept {
  return make_float3(left.x - right.x, left.y - right.y, left.z - right.z);
}

__host__ __device__ inline float3 scale(float3 value, float factor) noexcept {
  return make_float3(value.x * factor, value.y * factor, value.z * factor);
}

__host__ __device__ inline float dot(float3 left, float3 right) noexcept {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

__host__ __device__ inline float length_squared(float3 value) noexcept { return dot(value, value); }

__host__ __device__ inline float length(float3 value) noexcept {
  return sqrtf(length_squared(value));
}

__host__ __device__ inline float3 normalize_or_zero(float3 value) noexcept {
  const float squared_length = length_squared(value);
  if (squared_length == 0.0f) {
    return uniform(0.0f);
  }
  return scale(value, 1.0f / sqrtf(squared_length));
}

}  // namespace float3_ops
}  // namespace kernel_helper
