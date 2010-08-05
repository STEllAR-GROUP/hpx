# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Multilocality tests
set(TESTS
    cpi_test
   )
set(TESTS_dir ${hpx_BINARY_DIR}/tests/multilocality_tests)

foreach(test ${TESTS})
  set(${test}_SOURCES ${test}.cpp)
  add_hpx_executable(${test} SOURCES ${${test}_SOURCES})
  add_hpx_test_sweep(
      ${TESTS_dir}/${test} 
      SWEEP_APP --throws 10000000 
      SWEEP_HPX -s)
endforeach()
