# Copyright (c)      2018 Mikael Simberg
# Copyright (c) 2007-2015 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests spmd_block)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests ${tests} task_block task_block_executor task_block_par)
endif()

set(task_block_FLAGS DEPENDENCIES iostreams_component)
set(task_block_executor_FLAGS DEPENDENCIES iostreams_component)
set(task_block_par_FLAGS DEPENDENCIES iostreams_component)

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Parallelism/Algorithms/Block"
  )

  add_hpx_unit_test("modules.algorithms" ${test} ${${test}_PARAMETERS})
endforeach()
