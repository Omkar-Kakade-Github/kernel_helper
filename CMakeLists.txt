cmake_minimum_required(VERSION 3.24)

project(
  kernel_helper
  VERSION 0.1.0
  DESCRIPTION "Header-only CUDA kernel support utilities"
  LANGUAGES CXX CUDA
)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

find_package(CUDAToolkit 12 REQUIRED)

add_library(kernel_helper INTERFACE)
add_library(kernel_helper::kernel_helper ALIAS kernel_helper)

target_compile_features(
  kernel_helper
  INTERFACE cxx_std_17 cuda_std_17
)
target_compile_options(
  kernel_helper
  INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
)
target_include_directories(
  kernel_helper
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(kernel_helper INTERFACE CUDA::cudart)

option(KERNEL_HELPER_BUILD_TESTS "Build kernel_helper tests" ${PROJECT_IS_TOP_LEVEL})
option(KERNEL_HELPER_BUILD_EXAMPLES "Build kernel_helper examples" ${PROJECT_IS_TOP_LEVEL})
option(KERNEL_HELPER_INSTALL "Generate kernel_helper installation rules" ${PROJECT_IS_TOP_LEVEL})

if(KERNEL_HELPER_BUILD_TESTS)
  enable_testing()
endif()

if(KERNEL_HELPER_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(KERNEL_HELPER_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(KERNEL_HELPER_INSTALL)
  install(
    TARGETS kernel_helper
    EXPORT kernel_helperTargets
  )
  install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )
  install(
    EXPORT kernel_helperTargets
    FILE kernel_helperTargets.cmake
    NAMESPACE kernel_helper::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/kernel_helper"
  )

  configure_package_config_file(
    cmake/kernel_helperConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_helperConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/kernel_helper"
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_helperConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
  )
  install(
    FILES
      "${CMAKE_CURRENT_BINARY_DIR}/kernel_helperConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/kernel_helperConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/kernel_helper"
  )
  install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/kernel_helper")
endif()
