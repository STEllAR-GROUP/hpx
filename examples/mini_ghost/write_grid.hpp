//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_WRITE_GRID_HPP
#define HPX_EXAMPLES_MINI_WRITE_GRID_HPP

#include <examples/mini_ghost/grid.hpp>

#include <string>

namespace mini_ghost {
    template <typename Real>
    void write_grid(grid<Real> const & g, std::string filename, std::size_t z = 1);
}

#endif
