
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/include/iostreams.hpp>

#include <vector>

#include "grid.hpp"
#include "row.hpp"

namespace jacobi
{
    grid::grid(std::size_t nx, std::size_t ny, double value)
        //: rows(ny)
    {
        std::vector<hpx::id_type> ids = hpx::new_<server::row[]>(
            hpx::default_layout(hpx::find_all_localities()), ny).get();

        rows.reserve(ny);
        std::vector<hpx::lcos::future<void> > init_futures;
        init_futures.reserve(ny);
        for (hpx::naming::id_type const& id : ids)
        {
            row r; r.id = id;
            init_futures.push_back(r.init(nx, value));
            rows.push_back(r);
        }

        hpx::wait_all(init_futures);
    }
}
