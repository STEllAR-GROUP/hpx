# Copyright (c) 2014-2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    copy_range
    copyif_range
    foreach_range
    foreach_range_projection
    generate_range
    is_heap_range
    is_heap_until_range
    max_element_range
    min_element_range
    minmax_element_range
    partition_range
    partition_copy_range
    remove_copy_range
    remove_copy_if_range
    replace_range
    replace_copy_range
    replace_copy_if_range
    replace_if_range
    reverse_range
    reverse_copy_range
    rotate_range
    rotate_copy_range
    sort_range
    transform_range
    transform_range_binary
    transform_range_binary2
    unique_copy_range
   )

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Unit/Parallel/ContainerAlgorithms")

  add_hpx_unit_test("parallel" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.parallel.container_algorithms.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parallel.container_algorithms
                              tests.unit.parallel.container_algorithms.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parallel.container_algorithms.${test}
                              ${test}_test_exe)
endforeach()
