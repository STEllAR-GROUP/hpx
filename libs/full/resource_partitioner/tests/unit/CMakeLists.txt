# Copyright (c) 2017 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests named_pool_executor resource_partitioner_info used_pus)

set(named_pool_executor_PARAMETERS THREADS_PER_LOCALITY 4)
set(resource_partitioner_info_PARAMETERS THREADS_PER_LOCALITY 4)
set(used_pus_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Full/ResourcePartitioner/"
  )

  add_hpx_unit_test(
    "modules.resource_partitioner" ${test} ${${test}_PARAMETERS}
  )

endforeach()
