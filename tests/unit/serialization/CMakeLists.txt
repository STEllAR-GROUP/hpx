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
    serialization_map
    serialization_list
    serialization_set
    serialization_simple
    serialization_smart_ptr
    serialization_unordered_map
    serialization_vector
    serialization_variant
    serialize_buffer
    zero_copy_serialization
)

add_subdirectory(polymorphic)

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

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.serialization.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.serialization
                              tests.unit.serialization.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.serialization.${test}
                              ${test}_test_exe)
endforeach()

if(HPX_WITH_COMPILE_ONLY_TESTS)
  # add compile time tests
  set(compile_tests
      fail_compile_non_serializable_class
      succeed_compile_empty_class
     )

  set(fail_compile_non_serializable_class_FLAGS FAILURE_EXPECTED)

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

    # add a custom target for this example
    add_hpx_pseudo_target(tests.unit.serialization.${compile_test})

    # make pseudo-targets depend on master pseudo-target
    add_hpx_pseudo_dependencies(tests.unit.serialization
                                tests.unit.serialization.${compile_test})

    # add dependencies to pseudo-target
    add_hpx_pseudo_dependencies(tests.unit.serialization.${compile_test}
                                "tests.unit.serialization.${compile_test}")

  endforeach()
endif()
