# Copyright (c) 2007-2015 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests ${tests} duplicate_id_registration_1596
            pass_by_value_id_type_action send_gid_keep_component_1624
  )
endif()

if(HPX_WITH_NETWORKING)
  set(tests ${tests} register_with_basename_1804)
  set(register_with_basename_1804_PARAMETERS LOCALITIES 2)
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/AGAS"
  )

  add_hpx_regression_test("agas" ${test} ${${test}_PARAMETERS})

endforeach()
