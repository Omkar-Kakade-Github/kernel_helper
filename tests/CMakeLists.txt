include(FetchContent)

find_package(GTest 1.17 QUIET)
if(NOT GTest_FOUND)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(kernel_helper_tests kernel_helper_tests.cu)
target_link_libraries(
  kernel_helper_tests
  PRIVATE kernel_helper::kernel_helper GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(kernel_helper_tests DISCOVERY_TIMEOUT 60)

if(KERNEL_HELPER_INSTALL)
  configure_file(
    install_consumer.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/install_consumer.cmake"
    @ONLY
  )
  add_test(
    NAME kernel_helper.install_consumer
    COMMAND
      "${CMAKE_COMMAND}"
      -P "${CMAKE_CURRENT_BINARY_DIR}/install_consumer.cmake"
  )
endif()

foreach(integration IN ITEMS subdirectory fetch_content)
  configure_file(
    source_consumer.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${integration}_consumer.cmake"
    @ONLY
  )
  add_test(
    NAME "kernel_helper.${integration}_consumer"
    COMMAND
      "${CMAKE_COMMAND}"
      -Dintegration=${integration}
      -P "${CMAKE_CURRENT_BINARY_DIR}/${integration}_consumer.cmake"
  )
endforeach()
