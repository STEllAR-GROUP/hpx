# Copyright (c) 2010-2011 Dylan Stark
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(hpx_MULTILOCALITY_TESTS
    startup_test
    cpi_test
   )

set(hpx_MULTILOCALITY_TEST_DIR
    ${hpx_BINARY_DIR}/tests/function/multilocality)

foreach(hpx_MULTILOCALITY_TEST ${hpx_MULTILOCALITY_TESTS})
  set(${hpx_MULTILOCALITY_TEST}_SOURCES ${hpx_MULTILOCALITY_TEST}.cpp)
  add_hpx_executable(${hpx_MULTILOCALITY_TEST} SOURCES ${${hpx_MULTILOCALITY_TEST}_SOURCES})
  add_dependencies(tests.function.multilocality ${hpx_MULTILOCALITY_TEST}) 
endforeach()

add_hpx_test_sweep(
    ${hpx_MULTILOCALITY_TEST_DIR}/startup_test
    SWEEP_HPX -s -c startup_test)

add_hpx_test_sweep(
    ${hpx_MULTILOCALITY_TEST_DIR}/cpi_test
    SWEEP_APP --throws 10000000 
    SWEEP_HPX -s -c cpi_test)

