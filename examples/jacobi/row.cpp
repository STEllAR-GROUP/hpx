
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "row.hpp"

namespace jacobi
{
    row::row(std::size_t nx, double init)
        : values(new double[nx])
    {
        // TODO: replace with hpx::for_each
        row_range range(values, 0, nx);
        BOOST_FOREACH(double & v, range)
        {
            v = init;
        }
    }
}
