
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "grid.hpp"
#include "row.hpp"

#include <hpx/components/remote_object/object.hpp>
#include <hpx/components/remote_object/distributed_new.hpp>
#include <hpx/lcos/future_wait.hpp>

namespace jacobi
{
    grid::grid(std::size_t nx, std::size_t ny, double value)
        : rows(ny)
    {
        std::vector<hpx::lcos::future<hpx::components::object<row> > >
            row_futures =
                hpx::components::distributed_new<row>(ny, nx, value);
        
        hpx::lcos::wait(
            row_futures
          , [&](std::size_t i, hpx::components::object<row> const & r)
            {
                rows[i] = r;
            }
        );
    }
}
