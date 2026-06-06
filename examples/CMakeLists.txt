set(kernel_helper_host_examples
    error_math
)
set(kernel_helper_gpu_examples
    memory
    kernel_primitives
    atomics
    collectives
)

foreach(example IN LISTS kernel_helper_host_examples kernel_helper_gpu_examples)
  set(target "kernel_helper_${example}_example")
  add_executable("${target}" "${example}.cu")
  target_link_libraries("${target}" PRIVATE kernel_helper::kernel_helper)

  if(KERNEL_HELPER_BUILD_TESTS)
    add_test(NAME "kernel_helper.example.${example}" COMMAND "${target}")
  endif()
endforeach()

if(KERNEL_HELPER_BUILD_TESTS)
  foreach(example IN LISTS kernel_helper_gpu_examples)
    set_tests_properties(
      "kernel_helper.example.${example}"
      PROPERTIES SKIP_RETURN_CODE 77
    )
  endforeach()
endif()

