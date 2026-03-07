# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests partitioned_vector_fewer_partitions_than_localities)

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_FLAGS DEPENDENCIES partitioned_vector_component)
  set(${test}_PARAMETERS ${${test}_PARAMETERS} LOCALITIES 3
                         THREADS_PER_LOCALITY 1
  )

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Regressions/Modules/Full/SegmentedAlgorithms")

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_regression_test(
    "modules.segmented_algorithms" ${test} ${${test}_PARAMETERS}
  )

endforeach()
