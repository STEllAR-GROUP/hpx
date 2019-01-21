# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if (NOT HPX_WITH_TESTS AND HPX_TOP_LEVEL)
  return()
endif()
if (NOT HPX_CONFIG_WITH_TESTS)
  message(STATUS "Tests for config disabled")
  return()
endif()

add_custom_target(tests.unit.config)
add_dependencies(tests.unit tests.unit.config)
add_subdirectory(unit)

add_custom_target(tests.regressions.config)
add_dependencies(tests.regressions tests.regressions.config)
add_subdirectory(regressions)
