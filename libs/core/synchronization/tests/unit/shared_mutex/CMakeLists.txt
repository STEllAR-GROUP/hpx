# Copyright (c) 2015 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests shared_mutex1 shared_mutex2)

set(shared_mutex1_PARAMETERS THREADS_PER_LOCALITY 4)
set(shared_mutex2_PARAMETERS THREADS_PER_LOCALITY 4)

set(shared_mutex1_FLAGS DEPENDENCIES PRIVATE hpx_dependencies_boost)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Core/Synchronization/SharedMutex"
  )

  add_hpx_unit_test(
    "modules.synchronization.shared_mutex" ${test} ${${test}_PARAMETERS}
  )

endforeach()
