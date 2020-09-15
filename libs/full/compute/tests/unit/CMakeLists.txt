# Copyright (c) 2019 John Biddiscombe
# Copyright (c) 2016 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests block_allocator)

if(HPX_WITH_DISTRIBUTED_RUNTIME AND HPX_WITH_SHARED_PRIORITY_SCHEDULER)
  set(tests ${tests} numa_allocator)
endif()

# NB. threads = -2 = threads = 'cores' NB. threads = -1 = threads = 'all'
set(numa_allocator_PARAMETERS
    THREADS_PER_LOCALITY
    -2
    ARGS
    --size=128
    --nb=512
    --tiles-per-domain=2
    --col-proc=1
    --row-proc=1
)
set(numa_allocator_FLAGS DEPENDENCIES iostreams_component DEPENDENCIES PRIVATE
                         Hwloc::hwloc
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
    FOLDER "Tests/Unit/Modules/Full/Compute/Host"
  )

  target_include_directories(${test}_test SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
  add_hpx_unit_test("modules.compute" ${test} ${${test}_PARAMETERS})

endforeach()
