# Copyright (c) 2007-2008 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(hpx_TEST_SUBDIRS 
    function
    #unit
    )

foreach(hpx_TEST_SUBDIR ${hpx_TEST_SUBDIRS})
  add_custom_target(tests.${hpx_TEST_SUBDIR})
  add_subdirectory(${hpx_TEST_SUBDIR})
  add_dependencies(tests tests.${hpx_TEST_SUBDIR})
endforeach()

