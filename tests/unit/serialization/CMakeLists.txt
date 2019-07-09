# Copyright (c) 2014 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    serialization_array
    serialization_valarray
    serialization_builtins
    serialization_complex
    serialization_custom_constructor
    serialization_deque
    serialization_list
    serialization_map
    serialization_optional
    serialization_set
    serialization_simple
    serialization_smart_ptr
    serialization_unordered_map
    serialization_vector
    serialization_partitioned_vector
    serialization_variant
    serialize_buffer
    serialize_with_incompatible_signature
    zero_copy_serialization
)

if(HPX_WITH_CXX17_STRUCTURED_BINDINGS)
    set(tests ${tests}
        serialization_brace_initializable
)
endif()

add_subdirectory(polymorphic)

set(serialization_partitioned_vector_FLAGS DEPENDENCIES partitioned_vector_component)
set(serialization_partitioned_vector_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Unit/Serialization/")

  add_hpx_unit_test("serialization" ${test} ${${test}_PARAMETERS})

endforeach()

if(HPX_WITH_COMPILE_ONLY_TESTS)
  # add compile time tests
  set(compile_tests
      succeed_compile_empty_class
     )

  if(HPX_WITH_FAIL_COMPILE_TESTS)
    set(fail_compile_tests
       )
    foreach(fail_compile_test ${fail_compile_tests})
      set(${fail_compile_test}_FLAGS FAILURE_EXPECTED)
    endforeach()

    set(compile_tests ${compile_tests} ${fail_compile_tests})
  endif()

  foreach(compile_test ${compile_tests})
    set(sources
        ${compile_test}.cpp)

    source_group("Source Files" FILES ${sources})

    add_hpx_unit_compile_test(
      "serialization"
      ${compile_test}
      SOURCES ${sources}
      ${${compile_test}_FLAGS}
      FOLDER "Tests/Unit/Serialization/CompileOnly")

  endforeach()

endif()
