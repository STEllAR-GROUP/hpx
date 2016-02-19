//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_UNORDERED_MAP_HPP
#define HPX_SERIALIZATION_UNORDERED_MAP_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/serialization/map.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <unordered_map>

#include <boost/mpl/and.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/add_reference.hpp>

namespace hpx
{
    namespace serialization
    {
        template <class Key, class Value, class Hash, class KeyEqual, class Alloc>
        void serialize(input_archive& ar,
            std::unordered_map<Key, Value, Hash, KeyEqual, Alloc>& t, unsigned)
        {
            typedef std::unordered_map<Key, Value, Hash, KeyEqual, Alloc>
                container_type;

            typedef typename container_type::size_type size_type;
            typedef typename container_type::value_type value_type;

            size_type size;
            ar >> size; //-V128

            t.clear();
            for (size_type i = 0; i < size; ++i)
            {
                value_type v;
                ar >> v;
                t.insert(t.end(), std::move(v));
            }
        }

        template <class Key, class Value, class Hash, class KeyEqual, class Alloc>
        void serialize(output_archive& ar,
            const std::unordered_map<Key, Value, Hash, KeyEqual, Alloc>& t, unsigned)
        {
            typedef std::unordered_map<Key, Value, Hash, KeyEqual, Alloc>
                container_type;

            typedef typename container_type::value_type value_type;

            ar << t.size(); //-V128
            for(const value_type& val : t)
            {
                ar << val;
            }
        }
    }
}

#endif
