# Copyright (c) 2017 Ajai V George
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    partitioned_vector_copy
    partitioned_vector_for_each
    partitioned_vector_handle_values
    partitioned_vector_iter
    partitioned_vector_move
    partitioned_vector_target
    partitioned_vector_transform
    partitioned_vector_transform_reduce
    partitioned_vector_fill
    partitioned_vector_inclusive_scan
    partitioned_vector_exclusive_scan
    partitioned_vector_transform_scan
    partitioned_vector_reduce
    partitioned_vector_find
    partitioned_vector_find_end
    partitioned_vector_find_first_of
   )

# add dependencies to partitioned_vector_target when Cuda is enabled
if(HPX_WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})

  set(partitioned_vector_target_FLAGS
      DEPENDENCIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  set(${test}_FLAGS DEPENDENCIES partitioned_vector_component)
  set(${test}_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Parallel/SegmentedAlgorithms")

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER ${folder_name})

  add_hpx_unit_test("parallel" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.parallel.segmented_algorithms.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parallel.segmented_algorithms
                              tests.unit.parallel.segmented_algorithms.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parallel.segmented_algorithms.${test}
                              ${test}_test_exe)
endforeach()
