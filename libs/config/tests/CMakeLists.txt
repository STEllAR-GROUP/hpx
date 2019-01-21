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

if (HPX_WITH_TESTS_HEADERS)
  add_hpx_lib_header_tests(config)
endif()
