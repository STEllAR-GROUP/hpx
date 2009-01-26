//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MANDELBROT_JANUARY_23_2009_1020AM)
#define HPX_MANDELBROT_JANUARY_23_2009_1020AM

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_COMPONENT_EXPORT int mandelbrot(double x, double y, int iterations);

typedef 
    hpx::actions::plain_result_action3<int, double, double, int, mandelbrot> 
mandelbrot_action;

#endif
