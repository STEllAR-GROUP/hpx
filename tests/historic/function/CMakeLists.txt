# Copyright (c) 2007-2008 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(hpx_FUNCTION_TEST_SUBDIRS 
#    action_manager
#    applier
#    multilocality
    )

foreach(hpx_FUNCTION_TEST_SUBDIR ${hpx_FUNCTION_TEST_SUBDIRS})
  add_hpx_pseudo_target(tests.function.${hpx_FUNCTION_TEST_SUBDIR})
  add_subdirectory(${hpx_FUNCTION_TEST_SUBDIR})
  add_hpx_pseudo_dependencies(tests.function tests.function.${hpx_FUNCTION_TEST_SUBDIR})
endforeach()

