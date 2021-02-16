#  Copyright (c) 2014-2016 Hartmut Kaiser
#  Copyright (c) 2011 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    is_iterator_partitioned_vector
    partitioned_vector_view
    partitioned_vector_view_iterator
    partitioned_vector_subview
    coarray
    coarray_all_reduce
    serialization_partitioned_vector
)

set(is_iterator_partitioned_vector_FLAGS COMPONENT_DEPENDENCIES
                                         partitioned_vector
)
set(is_iterator_partitioned_vector_PARAMETERS THREADS_PER_LOCALITY 4)

set(partitioned_vector_view_FLAGS COMPONENT_DEPENDENCIES partitioned_vector)
set(partitioned_vector_view_PARAMETERS THREADS_PER_LOCALITY 4)

set(partitioned_vector_view_iterator_FLAGS COMPONENT_DEPENDENCIES
                                           partitioned_vector
)
set(partitioned_vector_view_iterator_PARAMETERS THREADS_PER_LOCALITY 4)

set(partitioned_vector_subview_FLAGS COMPONENT_DEPENDENCIES partitioned_vector)
set(partitioned_vector_subview_PARAMETERS THREADS_PER_LOCALITY 4)

set(coarray_FLAGS COMPONENT_DEPENDENCIES partitioned_vector)
set(coarray_PARAMETERS THREADS_PER_LOCALITY 4)

set(coarray_all_reduce_FLAGS COMPONENT_DEPENDENCIES partitioned_vector)
set(coarray_all_reduce_PARAMETERS THREADS_PER_LOCALITY 4)

set(serialization_partitioned_vector_FLAGS COMPONENT_DEPENDENCIES
                                           partitioned_vector
)
set(serialization_partitioned_vector_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Components/Containers/PartitionedVector")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_unit_test(
    "components.partitioned_vector" ${test} ${${test}_PARAMETERS}
  )
endforeach()
