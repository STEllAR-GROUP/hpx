# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(group performance.core)

set(benchmarks
    #measure_bind 
    #measure_getprefix 
    #measure_getrange 
    #measure_queryid
    #measure_registerid 
    #measure_resolve 
    #measure_unbind 
    #measure_unregister
    measure_thread_execution)

foreach(benchmark ${benchmarks})
  add_hpx_executable(${group}.${benchmark}_benchmark SOURCES ${benchmark}.cpp)
  add_hpx_pseudo_dependencies(${group} ${group}.${benchmark}_benchmark_exe)
endforeach()

