# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    libcds_deque libcds_flat_combining_priority_queue_stdthread
    libcds_map_hazard_pointer_hpxthread libcds_map_hazard_pointer_stdthread
)

# make sure we set the concurrent thread limit high enough
set(libcds_map_hazard_pointer_hpxthread_PARAMETERS
    THREADS_PER_LOCALITY 4
    "-Ihpx.cds.num_concurrent_hazard_pointer_threads=256"
)

foreach(test ${tests})

  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Parallelism/LibCDS"
  )

  add_hpx_unit_test("modules.libcds" ${test} ${${test}_PARAMETERS})

endforeach()
