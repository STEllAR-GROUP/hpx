//  Copyright (c) 2015 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_SET_HPP
#define HPX_SERIALIZATION_SET_HPP

#include <hpx/runtime/serialization/serialize.hpp>

#include <set>

namespace hpx { namespace serialization
{
    template <typename T, typename Allocator>
    void serialize(input_archive & ar, std::set<T, Allocator> & set, unsigned)
    {
        set.clear();
        std::size_t size;
        ar >> size;
        for (std::size_t i = 0; i < size; ++i) {
            T t;
            ar >> t;
            set.insert(t);
        }
    }

    template <typename T, typename Allocator>
    void serialize(output_archive & ar, std::set<T, Allocator> & set, unsigned)
    {
        ar << set.size(); //-V128
        if(set.empty()) return;
        for (typename std::set<T, Allocator>::iterator i = set.begin(); i != set.end(); ++i) {
            ar << *i;
        }
    }

}}

#endif
