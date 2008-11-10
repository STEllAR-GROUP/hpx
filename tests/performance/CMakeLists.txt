# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Performance measurement helpers
set(performance_TESTS 
    measure_bind 
    measure_getprefix 
    measure_getrange 
    measure_queryid
    measure_registerid 
    measure_resolve 
    measure_unbind 
    measure_unregister
    measure_thread_execution)

# build and install each of the measurement helpers
foreach(measure ${performance_TESTS})
    set(${measure}_SOURCES ${measure}.cpp high_resolution_timer.hpp)
    add_hpx_executable(${measure} SOURCES ${${measure}_SOURCES})
endforeach()

