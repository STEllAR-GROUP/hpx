//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include "mandelbrot.hpp"

///////////////////////////////////////////////////////////////////////////////
inline long double sqr(long double x)
{
    return x * x;
}

///////////////////////////////////////////////////////////////////////////////
int mandelbrot(double xpt, double ypt, int iterations)
{
    long double x = 0;
    long double y = 0;      //converting from pixels to points

    int k = 0;
    for(/**/; k <= iterations; ++k)
    {
        // The Mandelbrot Function Z = Z*Z+c into x and y parts
        long double xnew = sqr(x) - sqr(y) + xpt;
        long double ynew = 2 * x*y - ypt;
        if (sqr(xnew) + sqr(ynew) > 4) 
            break;
        x = xnew;
        y = ynew;
    }

    return (k >= iterations) ? 0 : k;
}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(mandelbrot_action);

