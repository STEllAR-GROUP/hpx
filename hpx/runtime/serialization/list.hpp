//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_LIST_HPP
#define HPX_SERIALIZATION_LIST_HPP

#include <hpx/runtime/serialization/serialize.hpp>

#include <list>

namespace hpx { namespace serialization
{
    template <typename T, typename Allocator>
    void serialize(input_archive & ar, std::list<T, Allocator> & ls, unsigned)
    {
        // normal load ...
        typedef typename std::list<T, Allocator>::size_type size_type;
        size_type size;
        ar >> size; //-V128
        if(size == 0) return;

        ls.resize(size);
        for(auto & li: ls)
        {
            ar >> li;
        }
    }

    template <typename T, typename Allocator>
    void serialize(output_archive & ar, const std::list<T, Allocator> & ls, unsigned)
    {
        // normal save ...
        ar << ls.size(); //-V128
        if(ls.empty()) return;

        for(auto const & li: ls)
        {
            ar << li;
        }
    }
}}

#endif
