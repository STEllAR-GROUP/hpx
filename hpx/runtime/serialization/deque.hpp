//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_DEQUE_HPP
#define HPX_SERIALIZATION_DEQUE_HPP

#include <hpx/runtime/serialization/serialize.hpp>

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

        d.resize(size);
        if(size == 0) return;

        for(auto & e: d)
        {
            ar >> e;
        }
    }

    template <typename T, typename Allocator>
    void serialize(output_archive & ar, const std::deque<T, Allocator> & d, unsigned)
    {
        // normal save ...
        std::uint64_t size = d.size();
        ar << size;
        if(d.empty()) return;

        for(auto const & e: d)
        {
            ar << e;
        }
    }
}}

#endif
