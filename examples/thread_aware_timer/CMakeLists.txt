# Copyright (c) 2007-2012 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_THREAD_AWARE_TIMER_COMPATIBILITY AND HPX_WITH_DISTRIBUTED_RUNTIME)
  add_hpx_executable(
    thread_aware_timer INTERNAL_FLAGS
    SOURCES thread_aware_timer.cpp
    FOLDER "Examples/ThreadAwareTimer"
    DEPENDENCIES iostreams_component
  )

  add_hpx_example_target_dependencies("thread_aware_timer" thread_aware_timer)

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test(
      "thread_aware_timer" thread_aware_timer THREADS_PER_LOCALITY 4
    )
  endif()
endif()
