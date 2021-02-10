# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

add_subdirectory(components)

set(tests component_action_move_semantics make_continuation_1615
          plain_action_move_semantics return_non_default_constructible_2847
)

set(component_action_move_semantics_FLAGS DEPENDENCIES
                                          action_move_semantics_component
)
set(component_action_move_semantics_PARAMETERS LOCALITIES 2
                                               THREADS_PER_LOCALITY 1
)
set(plain_action_move_semantics_FLAGS DEPENDENCIES
                                      action_move_semantics_component
)
set(plain_action_move_semantics_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 1)
set(return_non_default_constructible_2847_PARAMETERS LOCALITIES 2)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/Actions"
  )

  add_hpx_regression_test("actions" ${test} ${${test}_PARAMETERS})
endforeach()
