# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(hpx_APPLIER_TESTS
    applier_test_client
    applier_test_server
   )

set(hpx_APPLIER_TEST_DIR
    ${hpx_BINARY_DIR}/tests/function/applier)

foreach(hpx_APPLIER_TEST ${hpx_APPLIER_TESTS})
  set(${hpx_APPLIER_TEST}_SOURCES ${hpx_APPLIER_TEST}.cpp)
  add_hpx_executable(${hpx_APPLIER_TEST}
      SOURCES ${${hpx_APPLIER_TEST}_SOURCES}
      DEPENDENCIES accumulator_component)
  add_dependencies(tests.function.applier ${hpx_APPLIER_TEST}) 
  add_hpx_test(${hpx_APPLIER_TEST_DIR}/${hpx_APPLIER_TEST} 
      TEST 1_1 HPX -l 1:1 -s)
endforeach()

