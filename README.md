# kernel_helper

`kernel_helper` is a header-only CUDA C++17 support library for common kernel
infrastructure. Version 0.1.0 supports CUDA 12 or newer, Linux x86_64, and
Ampere-class GPUs (`SM 8.0+`).

The library intentionally focuses on kernel foundations:

- checked CUDA runtime calls with typed exceptions
- move-only device memory ownership and checked copies
- one-dimensional launch and grid-stride indexing helpers
- host/device math and named `float3` operations
- warp metadata and floating-point atomic min/max
- CUB-backed warp and block reductions

Scans, segmented reductions, streams/events as owned resources, pinned or
unified memory, occupancy tuning, and whole-array algorithms are outside the
v0.1 scope.

## Requirements

- CMake 3.24+
- CUDA Toolkit 12+
- A C++17/CUDA C++17-compatible host compiler
- An `SM 8.0+` GPU for runtime use

CUDA 13 no longer targets several older GPU generations. The stable project
contract is therefore Ampere or newer even when building with CUDA 12.

## Use From CMake

Install the project:

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DKERNEL_HELPER_BUILD_TESTS=OFF
cmake --build build
cmake --install build --prefix /path/to/prefix
```

Consume the installed package:

```cmake
find_package(kernel_helper 0.1 REQUIRED)

add_executable(my_kernel main.cu)
target_link_libraries(my_kernel PRIVATE kernel_helper::kernel_helper)
set_target_properties(my_kernel PROPERTIES CUDA_ARCHITECTURES 80)
```

Include either the umbrella header or a focused header:

```cpp
#include <kernel_helper/kernel_helper.cuh>
// or
#include <kernel_helper/memory.cuh>
```

`kernel_helper` does not force architecture flags, fast math, relocatable
device code, or warning settings on consumers. Its CMake target enables
NVCC extended lambdas because `grid_stride_loop` is designed for
device-callable lambdas.

## Error Handling

`KERNEL_HELPER_CUDA_CHECK(expression)` throws `kernel_helper::cuda_error`,
which retains the CUDA status, expression, file, and line.

Kernel launch checking and synchronization are deliberately separate:

```cpp
kernel<<<grid, block>>>(arguments);
KERNEL_HELPER_CUDA_CHECK_LAUNCH();  // checks launch setup only
KERNEL_HELPER_CUDA_SYNCHRONIZE();   // waits and reports execution errors
```

`KERNEL_HELPER_CUDA_STREAM_SYNCHRONIZE(stream)` is the stream equivalent.
No helper silently synchronizes after a launch. Resource destructors never
throw.

## Launch And Indexing

```cpp
__global__ void scale(float* values, std::uint64_t count, float factor) {
  kernel_helper::grid_stride_loop(
      count, [=] __device__(std::uint64_t index) {
        values[index] *= factor;
      });
}

const auto config = kernel_helper::make_launch_config_1d(count, 256);
if (config) {
  scale<<<config.grid, config.block>>>(values, count, 2.0f);
  KERNEL_HELPER_CUDA_CHECK_LAUNCH();
}
```

A zero element count returns an empty launch configuration. It must not be
used in CUDA launch syntax. Non-empty configurations are checked against the
current device's block and one-dimensional grid limits.

## Memory

`device_buffer<T>` owns `cudaMalloc` memory and is movable but not copyable.
It provides `data()`, `size()`, `size_bytes()`, `reset()`, `release()`, and
`swap()`.

```cpp
std::vector<float> host(1024, 1.0f);
kernel_helper::device_buffer<float> device(host.size());

kernel_helper::copy_h2d(device, host.data(), host.size());
kernel_helper::zero_fill(device, 128, 256); // count, destination offset
kernel_helper::copy_d2h(host.data(), device, host.size());
```

Raw-pointer copy functions check byte-count overflow. `device_buffer`
overloads additionally check offsets and element counts. Asynchronous H2D and
D2H copies follow CUDA's normal host-memory lifetime and pinning requirements;
the caller must keep host storage alive until the operation completes.

## Collectives

Collectives are thin CUB wrappers. Every thread in the logical warp or block
must call the function. `valid_items` describes a contiguous valid prefix.
Warp results are valid only in lane 0; block results are valid only in thread
0.

```cpp
template <int BlockThreads>
__global__ void block_total(const float* input, int count, float* output) {
  __shared__
      kernel_helper::block_reduce_temp_storage<float, BlockThreads> storage;

  const int thread = static_cast<int>(threadIdx.x);
  const float value = thread < count ? input[thread] : 0.0f;
  const float total =
      kernel_helper::block_sum<float, BlockThreads>(value, storage, count);

  if (thread == 0) {
    *output = total;
  }
}
```

CUB temporary storage may be reused only after all participating threads have
passed an appropriate synchronization point. Generic `warp_reduce` and
`block_reduce` accept a device-callable binary operation. Sum, minimum, and
maximum convenience wrappers are also provided.

## Atomics

`atomic_min` and `atomic_max` overloads support `float` and `double` using
CAS. Incoming NaNs are ignored. If the stored value is NaN, the first finite
input replaces it.

## Build And Test

GoogleTest 1.17.0 is used for tests. An installed package is preferred;
otherwise CMake downloads the pinned release when tests are enabled.

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DKERNEL_HELPER_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

GPU tests skip cleanly when no CUDA device is visible. Release validation also
runs the GPU suite and Compute Sanitizer on a self-hosted Ampere-or-newer
runner.

## Legacy Sample

The original experimental header is preserved at
`legacy/old_sample.cuh`. It is unsupported, is not installed, and should not
be included by new code.

## License

MIT
