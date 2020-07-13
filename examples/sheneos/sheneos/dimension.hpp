//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/hpx.hpp>

#include <hpx/serialization.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace sheneos
{
    struct dimension
    {
        enum type {
            ye = 0,
            temp = 1,
            rho = 2,
            dim = 3
        };

        dimension() : offset_(0), count_(0), size_(0) {}
        dimension(std::size_t size) : offset_(0), count_(0), size_(size) {}

        std::size_t offset_;    ///< Dimension offset.
        std::size_t count_;     ///< Dimension count.
        std::size_t size_;      ///< Overall size of this dimension.
    };
}

///////////////////////////////////////////////////////////////////////////////
// Non-intrusive serialization support.
namespace hpx { namespace serialization
{
    HPX_COMPONENT_EXPORT void
    serialize(input_archive&, sheneos::dimension&, unsigned int const);

    HPX_COMPONENT_EXPORT void
    serialize(output_archive&, sheneos::dimension&, unsigned int const);
}}


