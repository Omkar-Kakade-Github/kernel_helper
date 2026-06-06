#include <cstdint>
#include <iostream>
#include <kernel_helper/kernel_helper.cuh>
#include <vector>

__global__ void square_values(float* values, std::uint64_t count) {
  kernel_helper::grid_stride_loop(
      count, [=] __device__(std::uint64_t index) { values[index] *= values[index]; });
}

int main() {
  std::vector<float> host{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  kernel_helper::device_buffer<float> device(host.size());
  kernel_helper::copy_h2d(device, host.data(), host.size());

  const auto config = kernel_helper::make_launch_config_1d(host.size());
  if (config) {
    square_values<<<config.grid, config.block>>>(device.data(), device.size());
    KERNEL_HELPER_CUDA_CHECK_LAUNCH();
    KERNEL_HELPER_CUDA_SYNCHRONIZE();
  }

  kernel_helper::copy_d2h(host.data(), device, host.size());
  for (float value : host) {
    std::cout << value << '\n';
  }
}
