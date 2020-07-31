# Copyright (c) 2014-2020 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add subdirectories
set(subdirs detail util)

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.unit.modules.algorithms.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(
    tests.unit.modules.algorithms tests.unit.modules.algorithms.${subdir}
  )
endforeach()

# add tests
set(tests
    adjacentdifference
    adjacentdifference_exception
    adjacentdifference_bad_alloc
    adjacentfind
    adjacentfind_exception
    adjacentfind_bad_alloc
    adjacentfind_binary
    adjacentfind_binary_exception
    adjacentfind_binary_bad_alloc
    all_of
    any_of
    copy
    copyif_random
    copyif_forward
    copyif_exception
    copyif_bad_alloc
    copyn
    count
    countif
    destroy
    destroyn
    distance
    equal
    equal_binary
    exclusive_scan
    exclusive_scan2
    exclusive_scan_exception
    exclusive_scan_bad_alloc
    exclusive_scan_validate
    fill
    filln
    find
    findend
    findfirstof
    findfirstof_binary
    findif
    findifnot
    findifnot_exception
    findifnot_bad_alloc
    foreach
    foreach_executors
    foreach_prefetching
    foreach_projection
    foreachn
    foreachn_exception
    foreachn_bad_alloc
    foreachn_projection
    foreachn_projection_exception
    foreachn_projection_bad_alloc
    for_loop
    for_loop_exception
    for_loop_induction
    for_loop_induction_async
    for_loop_n
    for_loop_n_strided
    for_loop_reduction
    for_loop_reduction_async
    for_loop_strided
    generate
    generaten
    is_heap
    is_heap_until
    includes
    inclusive_scan
    inplace_merge
    is_partitioned
    is_sorted
    is_sorted_until
    lexicographical_compare
    make_heap
    max_element
    merge
    min_element
    minmax_element
    mismatch
    mismatch_binary
    move
    none_of
    parallel_sort
    partition
    partition_copy
    reduce_
    reduce_by_key
    remove
    remove1
    remove2
    remove_if
    remove_if1
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
    sort
    sort_by_key
    sort_exceptions
    stable_partition
    stable_sort
    stable_sort_exceptions
    swapranges
    transform
    transform_binary
    transform_binary2
    transform_exclusive_scan
    transform_inclusive_scan
    transform_reduce
    transform_reduce_binary
    transform_reduce_binary_exception
    transform_reduce_binary_bad_alloc
    uninitialized_copy
    uninitialized_copyn
    uninitialized_default_construct
    uninitialized_default_constructn
    uninitialized_fill
    uninitialized_filln
    uninitialized_move
    uninitialized_moven
    uninitialized_value_construct
    uninitialized_value_constructn
    unique
    unique_copy
)

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Parallelism/Algorithms")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("modules.algorithms" ${test} ${${test}_PARAMETERS})
endforeach()
