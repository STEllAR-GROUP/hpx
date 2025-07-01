
//  Copyright (c) 2012 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include "row_range.hpp"
#include "server/row.hpp"

#include <hpx/include/lcos.hpp>
#include <hpx/include/naming.hpp>

#include <cstddef>

namespace jacobi {
    struct row
    {
        row() {}

        hpx::future<void> init(std::size_t nx, double value = 0.0)
        {
            return hpx::async<server::row::init_action>(id, nx, value);
        }

        hpx::id_type id;

        hpx::future<row_range> get(std::size_t begin, std::size_t end)
        {
            return hpx::async<server::row::get_action>(id, begin, end);
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & id;
        }
    };
}    // namespace jacobi

#endif
