//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_MULTI_ARRAY_HPP
#define HPX_SERIALIZATION_MULTI_ARRAY_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/array.hpp>

#include <boost/multi_array.hpp>

#include <cstddef>

namespace hpx { namespace serialization
{
    template <class T, std::size_t N, class Allocator>
    void load(input_archive& ar, boost::multi_array<T, N, Allocator>& marray, unsigned)
    {
        boost::array<std::size_t, N> shape;
        ar & shape;

        marray.resize(shape);
        ar & make_array(marray.data(), marray.num_elements());
    }

    template <class T, std::size_t N, class Allocator>
    void save(output_archive& ar, const boost::multi_array<T, N,
        Allocator>& marray, unsigned)
    {
        ar & make_array(marray.shape(), marray.num_dimensions());
        ar & make_array(marray.data(), marray.num_elements());
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(
            (template <class T, std::size_t N, class Allocator>),
            (boost::multi_array<T, N, Allocator>));
}}

#endif
