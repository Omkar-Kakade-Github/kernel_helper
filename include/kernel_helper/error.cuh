#pragma once

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace kernel_helper {

class cuda_error : public std::runtime_error {
 public:
  cuda_error(cudaError_t code, const char* expression, const char* file, int line)
      : std::runtime_error(make_message(code, expression, file, line)),
        code_(code),
        expression_(expression),
        file_(file),
        line_(line) {}

  cudaError_t code() const noexcept { return code_; }
  const std::string& expression() const noexcept { return expression_; }
  const std::string& file() const noexcept { return file_; }
  int line() const noexcept { return line_; }

 private:
  static std::string make_message(cudaError_t code, const char* expression, const char* file,
                                  int line) {
    std::ostringstream stream;
    stream << "CUDA call failed: " << expression << " at " << file << ':' << line << " ("
           << cudaGetErrorName(code) << ": " << cudaGetErrorString(code) << ')';
    return stream.str();
  }

  cudaError_t code_;
  std::string expression_;
  std::string file_;
  int line_;
};

inline void check_cuda(cudaError_t result, const char* expression, const char* file, int line) {
  if (result != cudaSuccess) {
    throw cuda_error(result, expression, file, line);
  }
}

inline void check_last_launch(const char* file, int line) {
  check_cuda(cudaGetLastError(), "cudaGetLastError()", file, line);
}

inline void synchronize_and_check(const char* file, int line) {
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", file, line);
}

inline void synchronize_and_check(cudaStream_t stream, const char* file, int line) {
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(stream)", file, line);
}

}  // namespace kernel_helper

#define KERNEL_HELPER_CUDA_CHECK(expression) \
  ::kernel_helper::check_cuda((expression), #expression, __FILE__, __LINE__)

#define KERNEL_HELPER_CUDA_CHECK_LAUNCH() ::kernel_helper::check_last_launch(__FILE__, __LINE__)

#define KERNEL_HELPER_CUDA_SYNCHRONIZE() ::kernel_helper::synchronize_and_check(__FILE__, __LINE__)

#define KERNEL_HELPER_CUDA_STREAM_SYNCHRONIZE(stream) \
  ::kernel_helper::synchronize_and_check((stream), __FILE__, __LINE__)
