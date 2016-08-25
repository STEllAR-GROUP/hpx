
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef JACOBI_ROW_HPP
#define JACOBI_ROW_HPP

#include "row_range.hpp"
#include "server/row.hpp"

#include <hpx/include/naming.hpp>
#include <hpx/include/lcos.hpp>

#include <cstddef>

namespace jacobi
{
    struct row
    {
        row() {}

        hpx::lcos::future<void> init(std::size_t nx, double value = 0.0)
        {
            return hpx::async<server::row::init_action>(id, nx, value);
        }

        hpx::naming::id_type id;

        hpx::lcos::future<row_range> get(std::size_t begin, std::size_t end)
        {
            return hpx::async<server::row::get_action>(id, begin, end);
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & id;
        }
    };
}

#endif
