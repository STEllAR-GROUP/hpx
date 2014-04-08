//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_PARTITION_HPP
#define HPX_EXAMPLES_MINI_GHOST_PARTITION_HPP

#include <examples/mini_ghost/grid.hpp>

namespace mini_ghost {
    template <typename Real>
    struct partition
    {
        typedef grid<Real> grid_type;

    private:
        std::size_t id_;
        std::array<grid_type, 2> grids_;

        global_sum<Real> sum_allreduce;
        Real total_source;
        Real flux_out;
    };
}

#endif
