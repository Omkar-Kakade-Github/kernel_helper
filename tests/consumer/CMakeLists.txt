cmake_minimum_required(VERSION 3.24)
project(kernel_helper_consumer LANGUAGES CXX CUDA)

find_package(kernel_helper 0.1 REQUIRED)

add_executable(kernel_helper_consumer main.cu)
target_link_libraries(
  kernel_helper_consumer
  PRIVATE kernel_helper::kernel_helper
)

