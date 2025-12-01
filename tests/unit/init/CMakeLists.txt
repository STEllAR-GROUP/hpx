# Copyright (c) 2025 Agustin Berge
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests manage_runtime)

if(NOT MSVC)
  set(tests ${tests} auto_wrap_main)
  set(auto_wrap_main_FLAGS NOLIBS DEPENDENCIES HPX::hpx HPX::auto_wrap_main)
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
    FOLDER "Tests/Unit/Init"
  )

  add_hpx_unit_test("init" ${test} ${${test}_PARAMETERS})

endforeach()
