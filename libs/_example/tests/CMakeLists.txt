# Copyright (c) 2018 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


if (NOT HPX_WITH_TESTS AND HPX_TOP_LEVEL)
  return()
endif()
if (NOT HPX__EXAMPLE_WITH_TESTS)
  message(STATUS "Tests for _example disabled")
  return()
endif()

add_subdirectory(unit)
add_subdirectory(regressions)
add_subdirectory(performance)
