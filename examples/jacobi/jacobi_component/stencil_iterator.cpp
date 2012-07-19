
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include "stencil_iterator.hpp"
#include "server/stencil_iterator.hpp"

namespace jacobi
{
    hpx::lcos::future<void> stencil_iterator::init(
        jacobi::row const & r
      , std::size_t y
      , std::size_t nx
      , std::size_t ny
      , std::size_t l
    )
    {
        BOOST_ASSERT(id);
        return hpx::async<server::stencil_iterator::init_action>(id, r, y, nx, ny, l);
    }
        
    hpx::lcos::future<void> stencil_iterator::setup_boundary(
        stencil_iterator const & top
      , stencil_iterator const & bottom
    )
    {
        BOOST_ASSERT(id);
        return
            hpx::async<server::stencil_iterator::setup_boundary_action>(
                id
              , top
              , bottom
            );
    }
        
    hpx::lcos::future<void> stencil_iterator::step()
    {
        BOOST_ASSERT(id);
        return
            hpx::async<server::stencil_iterator::step_action>(id);
    }
    
    hpx::lcos::future<row_range> stencil_iterator::get_range(std::size_t begin, std::size_t end)
    {
        BOOST_ASSERT(id);
        return hpx::async<server::stencil_iterator::get_range_action>(id, begin, end);
    }
}
