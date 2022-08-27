
//  Copyright (c) 2012 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include "server/stencil_iterator.hpp"
#include "stencil_iterator.hpp"

#include <cstddef>

namespace jacobi {
    hpx::future<void> stencil_iterator::init(jacobi::row const& r,
        std::size_t y, std::size_t nx, std::size_t ny, std::size_t l)
    {
        HPX_ASSERT(id);
        return hpx::async<server::stencil_iterator::init_action>(
            id, r, y, nx, ny, l);
    }

    hpx::future<void> stencil_iterator::setup_boundary(
        stencil_iterator const& top, stencil_iterator const& bottom)
    {
        HPX_ASSERT(id);
        HPX_ASSERT(top.id);
        HPX_ASSERT(bottom.id);
        return hpx::async<server::stencil_iterator::setup_boundary_action>(
            id, top, bottom);
    }

    hpx::future<void> stencil_iterator::step()
    {
        HPX_ASSERT(id);
        return hpx::async<server::stencil_iterator::step_action>(id);
    }

    hpx::future<jacobi::row> stencil_iterator::get(std::size_t idx) const
    {
        HPX_ASSERT(id);
        return hpx::async<server::stencil_iterator::get_action>(id, idx);
    }
}    // namespace jacobi
#endif
