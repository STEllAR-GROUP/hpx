//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_MULTI_ARRAY_HPP
#define HPX_SERIALIZATION_MULTI_ARRAY_HPP

#include <hpx/serialization/serialization_fwd.hpp>

#if defined(HPX_SERIALIZATION_HAVE_BOOST_TYPES)
#include <hpx/serialization/array.hpp>

#include <boost/multi_array.hpp>

#include <cstddef>

namespace hpx { namespace serialization {

    template <typename T, std::size_t N, typename Allocator>
    void load(input_archive& ar, boost::multi_array<T, N, Allocator>& marray,
        unsigned)
    {
        boost::array<std::size_t, N> shape;
        ar& shape;

        marray.resize(shape);
        ar& make_array(marray.data(), marray.num_elements());
    }

    template <typename T, std::size_t N, typename Allocator>
    void save(output_archive& ar,
        const boost::multi_array<T, N, Allocator>& marray, unsigned)
    {
        ar& make_array(marray.shape(), marray.num_dimensions());
        ar& make_array(marray.data(), marray.num_elements());
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(
        (template <typename T, std::size_t N, typename Allocator>),
        (boost::multi_array<T, N, Allocator>) );
}}    // namespace hpx::serialization

#endif
#endif
