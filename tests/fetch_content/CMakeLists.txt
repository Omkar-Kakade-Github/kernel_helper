cmake_minimum_required(VERSION 3.24)
project(kernel_helper_fetch_content_consumer LANGUAGES CXX CUDA)

if(NOT KERNEL_HELPER_SOURCE_DIR)
  message(FATAL_ERROR "KERNEL_HELPER_SOURCE_DIR is required")
endif()

include(FetchContent)
FetchContent_Declare(
  kernel_helper
  SOURCE_DIR "${KERNEL_HELPER_SOURCE_DIR}"
)
FetchContent_MakeAvailable(kernel_helper)

add_executable(kernel_helper_fetch_content_consumer ../consumer/main.cu)
target_link_libraries(
  kernel_helper_fetch_content_consumer
  PRIVATE kernel_helper::kernel_helper
)

