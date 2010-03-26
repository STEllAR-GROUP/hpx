# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# recurse into all components subdirectories and execute the scripts there
set(quickstart_SUDIRS 
    #mandelbrot
   )

foreach(subdir ${quickstart_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

# define build targets for this directory
add_hpx_executable(fibonacci SOURCES fibonacci.cpp)
add_hpx_executable(fibonacci2 SOURCES fibonacci2.cpp)
add_hpx_executable(quicksort SOURCES quicksort.cpp)

