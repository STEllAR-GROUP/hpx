
//  Copyright (c) 2012 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "row_range.hpp"

#include <hpx/include/lcos.hpp>
#include <hpx/include/naming.hpp>

#include <cstddef>

namespace jacobi {
    struct row;

    struct HPX_COMPONENT_EXPORT stencil_iterator
    {
        stencil_iterator() {}

        ~stencil_iterator()
        { /*HPX_ASSERT(id);*/
        }

        hpx::future<void> init(jacobi::row const& r, std::size_t y_,
            std::size_t nx_, std::size_t ny_, std::size_t l);
        hpx::future<void> setup_boundary(
            stencil_iterator const& top, stencil_iterator const& bottom);

        hpx::future<void> step();

        hpx::future<jacobi::row> get(std::size_t idx) const;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & id;
        }

        hpx::id_type id;
    };
}    // namespace jacobi
