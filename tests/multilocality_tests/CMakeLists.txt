# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Multilocality tests
set(TESTS
    cpi_test
   )

foreach(test ${TESTS})
  set(${test}_SOURCES ${test}.cpp)
  add_hpx_executable(${test} SOURCES ${${test}_SOURCES})
endforeach()
