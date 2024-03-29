# Copyright (c) 2007-2021 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    channel
    client_then
    future_wait
    packaged_action
    promise
    promise_allocator
    promise_emplace
    use_allocator
)

set(future_wait_PARAMETERS THREADS_PER_LOCALITY 4)
set(packaged_action_PARAMETERS THREADS_PER_LOCALITY 4)
set(promise_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Full/LCOsDistributed")

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("modules.lcos_distributed" ${test} ${${test}_PARAMETERS})
endforeach()
