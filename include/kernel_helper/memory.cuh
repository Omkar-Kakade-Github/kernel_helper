#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <kernel_helper/error.cuh>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kernel_helper {

template <typename T>
constexpr std::size_t checked_byte_size(std::size_t count) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::length_error("CUDA allocation or copy byte count overflow");
  }
  return count * sizeof(T);
}

inline void check_range(std::size_t buffer_size, std::size_t offset, std::size_t count) {
  if (offset > buffer_size || count > buffer_size - offset) {
    throw std::out_of_range("CUDA buffer operation exceeds device_buffer bounds");
  }
}

template <typename T>
class device_buffer {
 public:
  device_buffer() noexcept = default;

  explicit device_buffer(std::size_t count) { allocate(count); }

  ~device_buffer() noexcept {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  device_buffer(const device_buffer&) = delete;
  device_buffer& operator=(const device_buffer&) = delete;

  device_buffer(device_buffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

  device_buffer& operator=(device_buffer&& other) noexcept {
    if (this != &other) {
      device_buffer temporary(std::move(other));
      swap(temporary);
    }
    return *this;
  }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  std::size_t size_bytes() const noexcept { return size_ * sizeof(T); }
  bool empty() const noexcept { return size_ == 0; }
  explicit operator bool() const noexcept { return data_ != nullptr; }

  void reset(std::size_t count = 0) {
    device_buffer replacement(count);
    swap(replacement);
  }

  T* release() noexcept {
    T* released = data_;
    data_ = nullptr;
    size_ = 0;
    return released;
  }

  void swap(device_buffer& other) noexcept {
    using std::swap;
    swap(data_, other.data_);
    swap(size_, other.size_);
  }

 private:
  void allocate(std::size_t count) {
    if (count == 0) {
      return;
    }
    const std::size_t bytes = checked_byte_size<T>(count);
    KERNEL_HELPER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), bytes));
    size_ = count;
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

template <typename T>
void swap(device_buffer<T>& left, device_buffer<T>& right) noexcept {
  left.swap(right);
}

template <typename T>
void copy_h2d(T* destination, const T* source, std::size_t count) {
  if (count == 0) {
    return;
  }
  KERNEL_HELPER_CUDA_CHECK(
      cudaMemcpy(destination, source, checked_byte_size<T>(count), cudaMemcpyHostToDevice));
}

template <typename T>
void copy_d2h(T* destination, const T* source, std::size_t count) {
  if (count == 0) {
    return;
  }
  KERNEL_HELPER_CUDA_CHECK(
      cudaMemcpy(destination, source, checked_byte_size<T>(count), cudaMemcpyDeviceToHost));
}

template <typename T>
void copy_d2d(T* destination, const T* source, std::size_t count) {
  if (count == 0) {
    return;
  }
  KERNEL_HELPER_CUDA_CHECK(
      cudaMemcpy(destination, source, checked_byte_size<T>(count), cudaMemcpyDeviceToDevice));
}

template <typename T>
void copy_h2d(device_buffer<T>& destination, const T* source, std::size_t count,
              std::size_t destination_offset = 0) {
  check_range(destination.size(), destination_offset, count);
  copy_h2d(destination.data() + destination_offset, source, count);
}

template <typename T>
void copy_d2h(T* destination, const device_buffer<T>& source, std::size_t count,
              std::size_t source_offset = 0) {
  check_range(source.size(), source_offset, count);
  copy_d2h(destination, source.data() + source_offset, count);
}

template <typename T>
void copy_d2d(device_buffer<T>& destination, const device_buffer<T>& source, std::size_t count,
              std::size_t destination_offset = 0, std::size_t source_offset = 0) {
  check_range(destination.size(), destination_offset, count);
  check_range(source.size(), source_offset, count);
  copy_d2d(destination.data() + destination_offset, source.data() + source_offset, count);
}

template <typename T>
void copy_h2d_async(T* destination, const T* source, std::size_t count,
                    cudaStream_t stream = nullptr) {
  if (count == 0) {
    return;
  }
  KERNEL_HELPER_CUDA_CHECK(cudaMemcpyAsync(destination, source, checked_byte_size<T>(count),
                                           cudaMemcpyHostToDevice, stream));
}

template <typename T>
void copy_d2h_async(T* destination, const T* source, std::size_t count,
                    cudaStream_t stream = nullptr) {
  if (count == 0) {
    return;
  }
  KERNEL_HELPER_CUDA_CHECK(cudaMemcpyAsync(destination, source, checked_byte_size<T>(count),
                                           cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void copy_d2d_async(T* destination, const T* source, std::size_t count,
                    cudaStream_t stream = nullptr) {
  if (count == 0) {
    return;
  }
  KERNEL_HELPER_CUDA_CHECK(cudaMemcpyAsync(destination, source, checked_byte_size<T>(count),
                                           cudaMemcpyDeviceToDevice, stream));
}

template <typename T>
void copy_h2d_async(device_buffer<T>& destination, const T* source, std::size_t count,
                    cudaStream_t stream = nullptr, std::size_t destination_offset = 0) {
  check_range(destination.size(), destination_offset, count);
  copy_h2d_async(destination.data() + destination_offset, source, count, stream);
}

template <typename T>
void copy_d2h_async(T* destination, const device_buffer<T>& source, std::size_t count,
                    cudaStream_t stream = nullptr, std::size_t source_offset = 0) {
  check_range(source.size(), source_offset, count);
  copy_d2h_async(destination, source.data() + source_offset, count, stream);
}

template <typename T>
void copy_d2d_async(device_buffer<T>& destination, const device_buffer<T>& source,
                    std::size_t count, cudaStream_t stream = nullptr,
                    std::size_t destination_offset = 0, std::size_t source_offset = 0) {
  check_range(destination.size(), destination_offset, count);
  check_range(source.size(), source_offset, count);
  copy_d2d_async(destination.data() + destination_offset, source.data() + source_offset, count,
                 stream);
}

template <typename T>
void zero_fill(T* destination, std::size_t count) {
  if (count == 0) {
    return;
  }
  KERNEL_HELPER_CUDA_CHECK(cudaMemset(destination, 0, checked_byte_size<T>(count)));
}

template <typename T>
void zero_fill(device_buffer<T>& destination, std::size_t count,
               std::size_t destination_offset = 0) {
  check_range(destination.size(), destination_offset, count);
  zero_fill(destination.data() + destination_offset, count);
}

template <typename T>
void zero_fill_async(T* destination, std::size_t count, cudaStream_t stream = nullptr) {
  if (count == 0) {
    return;
  }
  KERNEL_HELPER_CUDA_CHECK(cudaMemsetAsync(destination, 0, checked_byte_size<T>(count), stream));
}

template <typename T>
void zero_fill_async(device_buffer<T>& destination, std::size_t count,
                     cudaStream_t stream = nullptr, std::size_t destination_offset = 0) {
  check_range(destination.size(), destination_offset, count);
  zero_fill_async(destination.data() + destination_offset, count, stream);
}

}  // namespace kernel_helper
