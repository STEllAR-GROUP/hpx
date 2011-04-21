# Copyright (c) 2011 Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(group performance.allocator)

set(benchmarks
    malloc_then_free_constant_object_size
    new_then_delete_constant_object_size
    malloc_then_free_random_object_size
    new_then_delete_random_object_size
    malloc_and_free_constant_object_size
    new_and_delete_constant_object_size
    malloc_and_free_random_object_size
    new_and_delete_random_object_size)

set(allocators
    system)

if(TCMALLOC_FOUND)
  set(allocators ${allocators} tcmalloc)
endif()

if(JEMALLOC_FOUND)
  set(allocators ${allocators} jemalloc)
endif()

foreach(allocator ${allocators})
  string(TOUPPER ${allocator} name)

  foreach(benchmark ${benchmarks})
    add_hpx_executable(${group}.${allocator}.${benchmark}_benchmark
      MODULE ${group}.${allocator}
      SOURCES ${benchmark}.cpp
      DEPENDENCIES ${BOOST_FOUND_LIBRARIES} ${${name}_LIBRARY}
      NOLIBS)

    # depend on the executable
    add_hpx_pseudo_dependencies(${group}
      ${group}.${allocator}.${benchmark}_benchmark_exe)
  endforeach()
endforeach()

