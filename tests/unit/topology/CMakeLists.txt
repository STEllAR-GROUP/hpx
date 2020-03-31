# Copyright (c) 2019 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if (HPX_WITH_SHARED_PRIORITY_SCHEDULER)
  set(tests ${tests}
    numa_allocator
  )

  # NB. threads = -2 = threads = 'cores'
  # NB. threads = -1 = threads = 'all'
  set(numa_allocator_PARAMETERS
    THREADS_PER_LOCALITY -2
    ARGS --size=128 --nb=512 --tiles-per-domain=2 --col-proc=1 --row-proc=1)
endif()
set(numa_allocator_FLAGS
  COMPONENT_DEPENDENCIES iostreams
  DEPENDENCIES PRIVATE Hwloc::hwloc)

###############################################################################
foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Topology/")

  add_hpx_unit_test("topology" ${test} ${${test}_PARAMETERS})

endforeach()
