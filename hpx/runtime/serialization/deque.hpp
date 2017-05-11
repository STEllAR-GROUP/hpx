//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_DEQUE_HPP
#define HPX_SERIALIZATION_DEQUE_HPP

#include <hpx/runtime/serialization/serialize.hpp>

#include <deque>

namespace hpx { namespace serialization
{
    template <typename T, typename Allocator>
    void serialize(input_archive & ar, std::deque<T, Allocator> & d, unsigned)
    {
        // normal load ...
        typedef typename std::deque<T, Allocator>::size_type size_type;
        size_type size;
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
        ar << d.size(); //-V128
        if(d.empty()) return;

        for(auto const & e: d)
        {
            ar << e;
        }
    }
}}

#endif
