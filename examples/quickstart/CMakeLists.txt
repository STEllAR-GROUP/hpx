# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# recurse into all components subdirectories and execute the scripts there
set(hpx_QUICKSTART_SUBDIRS 
    mandelbrot)

foreach(hpx_QUICKSTART_SUBDIR ${hpx_QUICKSTART_SUBDIRS})
  add_subdirectory(${hpx_QUICKSTART_SUBDIR})
endforeach()

# add pesudo-targets
add_custom_target(examples.quickstart.fibonacci)
add_custom_target(examples.quickstart.fibonacci2)
add_custom_target(examples.quickstart.fibonacci2_memo)
add_custom_target(examples.quickstart.fibonacci2_lazy_future)
add_custom_target(examples.quickstart.fibonacci3)
add_custom_target(examples.quickstart.fibonacci4)
add_custom_target(examples.quickstart.fibonacci4_phased)
add_custom_target(examples.quickstart.quicksort)

# define build targets for this directory
add_hpx_executable(fibonacci SOURCES fibonacci.cpp)
add_hpx_executable(fibonacci2 SOURCES fibonacci2.cpp)
add_hpx_executable(fibonacci2_memo SOURCES fibonacci2_memo.cpp)
add_hpx_executable(fibonacci2_lazy_future SOURCES fibonacci2_lazy_future.cpp)
add_hpx_executable(fibonacci3 SOURCES fibonacci3.cpp)
add_hpx_executable(fibonacci4 SOURCES fibonacci4.cpp)
add_hpx_executable(fibonacci4_phased SOURCES fibonacci4_phased.cpp)
add_hpx_executable(quicksort SOURCES quicksort.cpp)

# add pseudo-target dependencies
add_dependencies(examples.quickstart.fibonacci fibonacci)
add_dependencies(examples.quickstart.fibonacci2 fibonacci2)
add_dependencies(examples.quickstart.fibonacci2_memo fibonacci2_memo)
add_dependencies(examples.quickstart.fibonacci2_lazy_future fibonacci2_lazy_future)
add_dependencies(examples.quickstart.fibonacci3 fibonacci3)
add_dependencies(examples.quickstart.fibonacci4 fibonacci4)
add_dependencies(examples.quickstart.fibonacci4_phased fibonacci4_phased)
add_dependencies(examples.quickstart.quicksort quicksort)

