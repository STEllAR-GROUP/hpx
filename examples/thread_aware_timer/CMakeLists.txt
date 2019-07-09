# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_executable(thread_aware_timer
                   SOURCES thread_aware_timer.cpp
                   FOLDER "Examples/ThreadAwareTimer"
                   DEPENDENCIES iostreams_component)

                 add_hpx_example_target_dependencies("thread_aware_timer" thread_aware_timer
                   DEPS_ONLY)

if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
  add_hpx_example_test("thread_aware_timer" thread_aware_timer THREADS_PER_LOCALITY 4)
endif()
