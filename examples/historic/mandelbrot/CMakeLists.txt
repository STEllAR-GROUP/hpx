# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# recurse into all components subdirectories and execute the scripts there
set(hpx_MANDELBROT_SUDIRS 
    mandelbrot_component)

foreach(hpx_MANDELBROT_SUBDIR ${hpx_MANDELBROT_SUBDIRS})
  add_subdirectory(${hpx_MANDELBROT_SUBDIR})
endforeach()

# add pesudo-target
add_hpx_pseudo_target(examples.quickstart.mandelbrot)

# define build targets for this directory
add_hpx_executable(mandelbrot 
  SOURCES mandelbrot.cpp
  DEPENDENCIES mandelbrot_component_component)

# add pseudo-target dependencies
add_hpx_pseudo_dependencies(examples.quickstart.mandelbrot mandelbrot_exe)

