# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(hpx_ACTION_MANAGER_TESTS
    am_test_client
    am_test_server
   )

set(hpx_ACTION_MANAGER_TEST_DIR
    ${hpx_BINARY_DIR}/tests/function/action_manager)

foreach(hpx_ACTION_MANAGER_TEST ${hpx_ACTION_MANAGER_TESTS})
  set(${hpx_ACTION_MANAGER_TEST}_SOURCES ${hpx_ACTION_MANAGER_TEST}.cpp)
  add_hpx_executable(${hpx_ACTION_MANAGER_TEST}
      SOURCES ${${hpx_ACTION_MANAGER_TEST}_SOURCES}
      DEPENDENCIES accumulator_component)
  add_dependencies(tests.function.action_manager ${hpx_ACTION_MANAGER_TEST}) 
  add_hpx_test(${hpx_ACTION_MANAGER_TEST_DIR}/${hpx_ACTION_MANAGER_TEST} 
      TEST 1_1 HPX -l 1:1 -s)
endforeach()

