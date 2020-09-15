# Copyright (c) 2019 Thomas Heller
# Copyright (c) 2020 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests basic_operation_state basic_receiver execution_context)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    NOLIBS
    DEPENDENCIES hpx_execution_base hpx_testing
                 ${BOOST_UNDERLYING_THREAD_LIBRARY}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Core/ExecutionBase"
  )

  add_hpx_unit_test("modules.execution_base" ${test} ${${test}_PARAMETERS})
  target_compile_definitions(${test}_test PRIVATE -DHPX_MODULE_STATIC_LINKING)

endforeach()
