# Copyright (c) 2019 The STE||AR-Group
#               2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests exception)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests ${tests} handled_exception unhandled_exception)
endif()

set(exception_LIBS NOLIBS DEPENDENCIES hpx_errors hpx_testing)
set(unhandled_exception_PARAMETERS FAILURE_EXPECTED)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Core/Errors")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS} ${${test}_LIBS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("modules.errors" ${test} ${${test}_PARAMETERS})
endforeach()
