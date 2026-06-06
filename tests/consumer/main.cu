#include <kernel_helper/kernel_helper.cuh>

__global__ void increment(int* value) {
  if (kernel_helper::global_thread_id_1d() == 0) {
    ++*value;
  }
}

int main() {
  static_assert(kernel_helper::ceil_div(7u, 3u) == 3u);
  return 0;
}
