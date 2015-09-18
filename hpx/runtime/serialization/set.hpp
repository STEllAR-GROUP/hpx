//  Copyright (c) 2015 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_SET_HPP
#define HPX_SERIALIZATION_SET_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <set>

namespace hpx { namespace serialization
{
    template <class T, class Compare, class Allocator>
    void serialize(input_archive & ar, std::set<T, Compare, Allocator> & set, unsigned)
    {
        set.clear();
        boost::uint64_t size;
        ar >> size;
        for (std::size_t i = 0; i < size; ++i) {
            T t;
            ar >> t;
            set.insert(set.end(), std::move(t));
        }
    }

    template <class T, class Compare, class Allocator>
    void serialize(output_archive & ar,const std::set<T, Compare, Allocator> & set,
        unsigned)
    {
        boost::uint64_t size = set.size();
        ar << size;
        if(set.empty()) return;
        for (T const& i: set) {
            ar << i;
        }
    }

}}

#endif
