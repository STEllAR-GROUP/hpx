# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if("$ENV{HPX_TESTS}" STREQUAL "all")
  set(test_SUBDIRS 
      #am_tests 
      #applier_tests 
      #component_tests 
      lco_tests 
      multilocality_tests
      #performance 
      #performance_counters 
      #threadmanager_tests
     )
  
  foreach(subdir ${test_SUBDIRS})
      add_subdirectory(${subdir})
  endforeach()
endif("$ENV{HPX_TESTS}" STREQUAL "all")

