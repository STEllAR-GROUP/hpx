# Copyright (c) 2014-2016 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

if(NOT HPX_WITH_STATIC_LINKING)
  set(_libraries hpx iostreams_component)

  add_library(
    test_server_1950 SHARED EXCLUDE_FROM_ALL server_1950.cpp server_1950.hpp
  )
  target_link_libraries(test_server_1950 PUBLIC ${_libraries})

  set_target_properties(
    test_server_1950 PROPERTIES FOLDER "Tests/Regressions/Build/"
  )

  add_executable(test_client_1950_test EXCLUDE_FROM_ALL client_1950.cpp)
  target_link_libraries(
    test_client_1950_test PRIVATE ${_libraries} test_server_1950
  )

  # Keep the same semantics *_test as the other tests
  set_target_properties(
    test_client_1950_test PROPERTIES FOLDER "Tests/Regressions/Build/"
  )

  add_hpx_regression_test(
    "build" test_client_1950 EXECUTABLE "$<TARGET_FILE:test_client_1950_test>"
    PSEUDO_DEPS_NAME test_client_1950
  )
endif()
