# Copyright (c) 2014-2016 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

include(HPX_CXXModules)

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

  if(HPX_WITH_CXX_MODULES)
    hpx_configure_module_consumer(test_server_1950 hpx_core_module_if)
    hpx_configure_module_consumer(test_client_1950_test hpx_core_module_if)
  endif()

  add_hpx_regression_test(
    "build" test_client_1950
    EXECUTABLE "$<TARGET_FILE:test_client_1950_test>"
    PSEUDO_DEPS_NAME test_client_1950
  )
endif()

# Smoke-test: 'import HPX.Core;' (local runtime)
if(HPX_WITH_CXX_MODULES)
  add_executable(use_cxx_module_test EXCLUDE_FROM_ALL use_cxx_module.cpp)
  target_link_libraries(use_cxx_module_test PRIVATE hpx)
  hpx_configure_module_consumer(use_cxx_module_test hpx_core_module_if)
  set_target_properties(
    use_cxx_module_test PROPERTIES FOLDER "Tests/Regressions/Build/"
  )
  add_hpx_regression_test(
    "build" use_cxx_module
    EXECUTABLE "$<TARGET_FILE:use_cxx_module_test>"
    PSEUDO_DEPS_NAME use_cxx_module
  )

  # Smoke-test: 'import HPX.Core.Full;' (distributed runtime)
  if(HPX_WITH_DISTRIBUTED_RUNTIME AND TARGET hpx_full_module_if)
    add_executable(
      use_cxx_module_full_test EXCLUDE_FROM_ALL use_cxx_module_full.cpp
    )
    target_link_libraries(use_cxx_module_full_test PRIVATE hpx)
    hpx_configure_module_consumer(use_cxx_module_full_test hpx_full_module_if)
    set_target_properties(
      use_cxx_module_full_test PROPERTIES FOLDER "Tests/Regressions/Build/"
    )
    add_hpx_regression_test(
      "build" use_cxx_module_full
      EXECUTABLE "$<TARGET_FILE:use_cxx_module_full_test>"
      PSEUDO_DEPS_NAME use_cxx_module_full
    )
  endif()
endif()
