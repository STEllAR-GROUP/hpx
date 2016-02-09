# Copyright (c) 2014-2016 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_STATIC_LINKING)
  set(_libraries hpx iostreams_component)
else()
  set(_libraries hpx hpx_init iostreams_component)
endif()

add_library(test_server_1950 SHARED server_1950.cpp)
target_link_libraries(test_server_1950 ${_libraries})

set_target_properties(test_server_1950
    PROPERTIES FOLDER "Tests/Regressions/Build/")

add_executable(test_client_1950 client_1950.cpp)
target_link_libraries(test_client_1950 ${_libraries} test_server_1950)

set_target_properties(test_client_1950
    PROPERTIES FOLDER "Tests/Regressions/Build/")

add_hpx_regression_test("build" test_client_1950
  EXECUTABLE "$<TARGET_FILE:test_client_1950>")

# add a custom target for this example
add_hpx_pseudo_target(tests.regressions.build.1950)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(tests.regressions.build_dir
  tests.regressions.build.1950)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(tests.regressions.build.1950
  test_client_1950)
