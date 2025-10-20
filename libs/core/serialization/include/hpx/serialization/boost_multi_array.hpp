//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/config/defines.hpp>

#if defined(HPX_SERIALIZATION_HAVE_BOOST_TYPES)
#include <hpx/serialization/array.hpp>
#include <hpx/serialization/macros.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <boost/multi_array.hpp>

#include <cstddef>

namespace hpx::serialization {

    HPX_CXX_EXPORT template <typename T, std::size_t N, typename Allocator>
    void load(input_archive& ar, boost::multi_array<T, N, Allocator>& marray,
        unsigned)
    {
        boost::array<std::size_t, N> shape;
        // clang-format off
        ar & shape;
        // clang-format on

        marray.resize(shape);
        // clang-format off
        ar & make_array(marray.data(), marray.num_elements());
        // clang-format on
    }

    HPX_CXX_EXPORT template <typename T, std::size_t N, typename Allocator>
    void save(output_archive& ar,
        boost::multi_array<T, N, Allocator> const& marray, unsigned)
    {
        // clang-format off
        ar & make_array(marray.shape(), marray.num_dimensions());
        ar & make_array(marray.data(), marray.num_elements());
        // clang-format on
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(HPX_CXX_EXPORT,
        (template <typename T, std::size_t N, typename Allocator>),
        (boost::multi_array<T, N, Allocator>) )
}    // namespace hpx::serialization

#endif
