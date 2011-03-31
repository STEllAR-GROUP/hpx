# Copyright (c) 2011 Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(type performance)

set(group allocator)

set(benchmarks
    malloc_then_free_constant_object_size
    new_then_delete_constant_object_size
    malloc_then_free_random_object_size
    new_then_delete_random_object_size
    malloc_and_free_constant_object_size
    new_and_delete_constant_object_size
    malloc_and_free_random_object_size
    new_and_delete_random_object_size
    )

foreach(benchmark ${benchmarks})
  add_hpx_executable(${type}.${group}.${benchmark}_benchmark
    MODULE ${type}.${group}
    SOURCES ${benchmark}.cpp)

  # depend on the executable
  add_dependencies(tests.${type}.${group}
                   ${type}.${group}.${benchmark}_benchmark_exe)
endforeach()

