
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef JACOBI_GRID_HPP
#define JACOBI_GRID_HPP

#include "row.hpp"

#include <cstddef>
#include <vector>

namespace jacobi
{
    struct row;

    struct HPX_COMPONENT_EXPORT grid
    {
        grid() {}
        grid(std::size_t nx, std::size_t ny, double value);

        typedef
            std::vector<row>
            rows_type;

        rows_type rows;

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & rows;
        }
    };
}

#endif
