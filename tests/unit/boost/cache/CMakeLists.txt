# Copyright (c) 2011 Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(group unit.boost.cache) 

set(testcases

CMakeLists.txt
    local_lru_cache
    local_mru_cache
    local_statistics)

foreach(testcase ${testcases})
  add_hpx_test(${group}.${testcase} SOURCES ${testcase}.cpp)

  # depend on the executable
  add_hpx_pseudo_dependencies(${group} ${group}.${testcase}_test)
endforeach()

