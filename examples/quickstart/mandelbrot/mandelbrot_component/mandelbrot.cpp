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
mandelbrot::result mandelbrot_func(mandelbrot::data const& data)
{
    long double x = (double(data.x_) / data.sizex_) * (data.maxx_ - data.minx_) + data.minx_;
    long double y = (double(data.y_) / data.sizey_) * (data.maxy_ - data.miny_) + data.miny_;

    long double xpt = x;
    long double ypt = y;

    boost::uint32_t k = 0;
    for(/**/; k < data.iterations_; ++k)
    {
        // The Mandelbrot Function Z = Z*Z+c into x and y parts
        long double xnew = sqr(x) - sqr(y) + xpt;
        long double ynew = 2 * x*y - ypt;
        if (sqr(xnew) + sqr(ynew) > 4) 
            break;
        x = xnew;
        y = ynew;
    }

    if (data.debug_)
    {
        std::cerr << "X: " << data.x_ << ", Y: " << data.y_ 
                  << ", result: " << ((k >= data.iterations_) ? 0 : k) 
                  << std::endl;
    }
    return mandelbrot::result(data.x_, data.y_, (k >= data.iterations_) ? 0 : k);
}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(mandelbrot_action);

