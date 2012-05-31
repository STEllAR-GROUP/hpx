
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef JACOBI_GRID_HPP
#define JACOBI_GRID_HPP

#include <hpx/components/dataflow/dataflow_object.hpp>

#include <vector>

namespace jacobi
{
    struct row;

    struct grid
    {
        grid(std::size_t nx, std::size_t ny, double value);

        typedef 
            std::vector<
                hpx::components::dataflow_object<row>
            >
            rows_type;

        rows_type rows;
    };
}

#endif
