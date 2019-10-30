# Copyright (c) 2013 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    mem_fn_derived_test
    mem_fn_dm_test
#    mem_fn_eq_test
    mem_fn_rv_test
    mem_fn_test
    mem_fn_unary_addr_test
    mem_fn_void_test
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
    FOLDER "Tests/Unit/Util/Mem_Fn")

  add_hpx_unit_test("util.mem_fn" ${test} ${${test}_PARAMETERS})

endforeach()
