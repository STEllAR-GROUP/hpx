# Copyright (c) 2013 Agustin Berge
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    bind_const_test
    bind_cv_test
    bind_dm2_test
    bind_dm3_test
    bind_dm_test
    bind_placeholder_test
    bind_rv_sp_test
    bind_rvalue_test
    bind_stateful_test
    bind_test
    protect_test
   )

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
    FOLDER "Tests/Unit/Util/Bind")

  add_hpx_unit_test("util.bind" ${test} ${${test}_PARAMETERS})

endforeach()
