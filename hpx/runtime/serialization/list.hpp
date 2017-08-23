//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_LIST_HPP
#define HPX_SERIALIZATION_LIST_HPP

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/detail/serialize_collection.hpp>

#include <cstdint>
#include <list>

namespace hpx { namespace serialization
{
    template <typename T, typename Allocator>
    void serialize(input_archive & ar, std::list<T, Allocator> & ls, unsigned)
    {
        // normal load ...
        std::uint64_t size;
        ar >> size; //-V128
        if(size == 0) return;

        detail::load_collection(ar, ls, size);
    }

    template <typename T, typename Allocator>
    void serialize(output_archive & ar, const std::list<T, Allocator> & ls, unsigned)
    {
        // normal save ...
        std::uint64_t size = ls.size();
        ar << size;
        if(ls.empty()) return;

        detail::save_collection(ar, ls);
    }
}}

#endif
