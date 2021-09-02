# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Function tests
set(function_tests bind_sfinae_5488 is_callable_1179 protect_with_nullary_pfo)

foreach(test ${function_tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  hpx_local_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    NOLIBS
    DEPENDENCIES hpx_local
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Regressions/Modules/Core/Functional"
  )

  hpx_local_add_regression_test(
    "modules.functional" ${test} ${${test}_PARAMETERS}
  )

endforeach()
