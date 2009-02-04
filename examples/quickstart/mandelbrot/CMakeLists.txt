# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# recurse into all components subdirectories and execute the scripts there
set(mandelbrot_SUDIRS 
    mandelbrot_component)

foreach(subdir ${mandelbrot_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

# define build targets for this directory
add_hpx_executable(mandelbrot 
    SOURCES mandelbrot.cpp
    DEPENDENCIES mandelbrot_component_component)

