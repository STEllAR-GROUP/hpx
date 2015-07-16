# Copyright (c) 2014-2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    adjacentdifference
    adjacentfind
    adjacentfind_binary
    all_of
    any_of
    copy
    copyif
    copyn
    count
    countif
    equal
    equal_binary
    exclusive_scan
    fill
    filln
    find
    findend
    findfirstof
    findfirstof_binary
    findif
    findifnot
    foreach
    foreach_executors
    foreachn
    generate
    generaten
    includes
    inclusive_scan
    inclusive_scan_executors
    inner_product
    is_partitioned
    is_sorted
    is_sorted_executors
    is_sorted_until
    lexicographical_compare
    max_element
    min_element
    minmax_element
    mismatch
    mismatch_binary
    move
    none_of
    reduce_
    remove_copy
    remove_copy_if
    replace
    replace_if
    replace_copy
    replace_copy_if
    reverse
    reverse_copy
    rotate
    rotate_copy
    search
    searchn
    set_difference
    set_intersection
    set_symmetric_difference
    set_union
    swapranges
    transform
    transform_binary
    transform_exclusive_scan
    transform_inclusive_scan
    transform_reduce
    uninitialized_copy
    uninitialized_copy_executors
    uninitialized_copyn
    uninitialized_fill
    uninitialized_filln
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
                     FOLDER "Tests/Unit/Parallel/Algorithms")

  add_hpx_unit_test("parallel" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.parallel.algorithms.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parallel.algorithms
                              tests.unit.parallel.algorithms.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parallel.algorithms.${test}
                              ${test}_test_exe)
endforeach()
