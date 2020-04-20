# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Threading tests
set(threading_tests
    condition_variable2
    condition_variable3
    condition_variable4
    jthread1
    jthread2
    stop_tokencb1
    stop_tokenrace
)

set(condition_variable2_PARAMETERS THREADS_PER_LOCALITY 4)
set(condition_variable3_PARAMETERS THREADS_PER_LOCALITY 4)
set(condition_variable4_PARAMETERS THREADS_PER_LOCALITY 4)
set(jthread1_PARAMETERS THREADS_PER_LOCALITY 4)
set(jthread2_PARAMETERS THREADS_PER_LOCALITY 4)
set(stop_tokencb1_PARAMETERS THREADS_PER_LOCALITY 4)
set(stop_tokenrace_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${threading_tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(${test}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Threading")

  add_hpx_unit_test("modules.threading" ${test})

endforeach()

