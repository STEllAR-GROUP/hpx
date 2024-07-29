# Copyright (c) 2014-2022 Hartmut Kaiser
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
    adjacentdifference_sender
    adjacentfind
    adjacentfind_binary
    adjacentfind_sender
    all_of
    all_of_sender
    any_of
    any_of_sender
    copy
    copy_sender
    copyif_random
    copyif_forward
    copyif_exception
    copyif_bad_alloc
    # copyif_sender
    copyn
    copyn_sender
    count
    count_sender
    countif
    countif_sender
    destroy
    destroy_sender
    destroyn
    destroyn_sender
    ends_with
    ends_with_sender
    equal
    equal_binary
    equal_sender
    equal_binary_sender
    exclusive_scan
    exclusive_scan2
    exclusive_scan_exception
    exclusive_scan_bad_alloc
    exclusive_scan_validate
    fill
    fill_sender
    filln
    filln_sender
    find
    find_sender
    findend
    findend_sender
    findfirstof
    findfirstof_binary
    findfirstof_sender
    findif
    findif_sender
    findifnot
    findifnot_sender
    foreach
    foreach_executors
    foreach_prefetching
    foreach_sender
    foreach_scheduler
    foreachn
    foreachn_exception
    foreachn_bad_alloc
    foreachn_sender
    for_loop
    for_loop_exception
    for_loop_induction
    for_loop_induction_async
    for_loop_n
    # for_loop_n_sender
    for_loop_n_strided
    # for_loop_n_strided_sender
    for_loop_reduction
    for_loop_reduction_async
    # for_loop_sender
    for_loop_strided
    # for_loop_strided_sender
    generate
    generate_sender
    generaten
    generaten_sender
    is_heap
    # is_heap_sender
    is_heap_until
    # is_heap_until_sender
    includes
    # includes_sender
    inclusive_scan
    inclusive_scan_exception
    inplace_merge
    # inplace_merge_sender
    is_partitioned
    is_partitioned_sender
    is_sorted
    is_sorted_sender
    is_sorted_until
    is_sorted_until_sender
    lexicographical_compare
    lexicographical_compare_sender
    make_heap
    # make_heap_sender
    max_element
    max_element_sender
    merge
    # merge_sender
    min_element
    min_element_sender
    minmax_element
    minmax_element_sender
    mismatch
    mismatch_binary
    # mismatch_sender
    move
    # move_sender
    nth_element
    # nth_element_sender
    none_of
    none_of_sender
    parallel_sort
    partial_sort
    # partial_sort_sender
    partial_sort_copy
    # partial_sort_copy_sender
    partition
    # partition_sender
    partition_copy
    reduce_
    reduce_sender
    reduce_by_key
    remove
    # remove_sender
    remove1
    remove2
    remove_if
    # remove_if_sender
    remove_if1
    remove_copy
    remove_copy_if
    replace
    replace_sender
    replace_if
    replace_if_sender
    replace_copy
    replace_copy_sender
    replace_copy_if
    replace_copy_if_sender
    reverse
    reverse_copy
    reverse_sender
    reverse_copy_sender
    rotate
    rotate_copy
    # rotate_sender rotate_copy_sender
    search
    # search_sender
    searchn
    # searchn_sender
    set_difference
    # set_difference_sender
    set_intersection
    # set_intersection_sender
    set_symmetric_difference
    # set_symmetric_difference_sender
    set_union
    # set_union_sender
    shift_left
    # shift_left_sender
    shift_right
    # shift_right_sender
    sort
    # sort_sender
    sort_by_key
    # sort_by_key_sender
    sort_exceptions
    stable_partition
    # stable_partition_sender
    stable_sort
    stable_sort_exceptions
    # stable_sort_sender
    starts_with
    # starts_with_sender
    swapranges
    # swapranges_sender
    transform
    transform_binary
    transform_binary2
    # transform_sender
    transform_exclusive_scan
    transform_inclusive_scan
    transform_reduce
    transform_reduce_binary
    transform_reduce_binary_exception
    transform_reduce_binary_bad_alloc
    # transform_reduce_sender
    uninitialized_copy
    uninitialized_copyn
    uninitialized_default_construct
    uninitialized_default_constructn
    uninitialized_fill
    uninitialized_filln
    uninitialized_move
    uninitialized_moven
    uninitialized_relocate_backward
    uninitialized_relocate
    uninitialized_relocaten
    uninitialized_value_construct
    uninitialized_value_constructn
    unique
    # unique_sender
    unique_copy
)

if(HPX_WITH_CXX17_STD_EXECUTION_POLICES)
  set(tests ${tests} foreach_std_policies)
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Core/Algorithms")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_unit_test(
    "modules.algorithms.algorithms" ${test} ${${test}_PARAMETERS}
  )
endforeach()
