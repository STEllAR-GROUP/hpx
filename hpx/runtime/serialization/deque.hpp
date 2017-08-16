//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_DEQUE_HPP
#define HPX_SERIALIZATION_DEQUE_HPP

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/detail/serialize_collection.hpp>

#include <cstdint>
#include <deque>

namespace hpx { namespace serialization
{
    template <typename T, typename Allocator>
    void serialize(input_archive & ar, std::deque<T, Allocator> & d, unsigned)
    {
        // normal load ...
        std::uint64_t size;
        ar >> size; //-V128
        if(size == 0) return;

        detail::load_collection(ar, d, size);
    }

    template <typename T, typename Allocator>
    void serialize(output_archive & ar, const std::deque<T, Allocator> & d, unsigned)
    {
        // normal save ...
        std::uint64_t size = d.size();
        ar << size;
        if(d.empty()) return;

        detail::save_collection(ar, d);
    }
}}

#endif
