# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(mandelbrot_HEADERS
    mandelbrot.hpp)

set(mandelbrot_SOURCES
    mandelbrot.cpp)

add_hpx_component(mandelbrot_component
    SOURCES ${mandelbrot_SOURCES} 
    HEADERS ${mandelbrot_HEADERS})

