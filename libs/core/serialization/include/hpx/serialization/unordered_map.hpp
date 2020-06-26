//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/map.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/serialize.hpp>

#include <unordered_map>
#include <utility>

namespace hpx { namespace serialization {

    template <typename Key, typename Value, typename Hash, typename KeyEqual,
        typename Alloc>
    void serialize(input_archive& ar,
        std::unordered_map<Key, Value, Hash, KeyEqual, Alloc>& t, unsigned)
    {
        using container_type =
            std::unordered_map<Key, Value, Hash, KeyEqual, Alloc>;

        using size_type = typename container_type::size_type;
        using value_type = typename container_type::value_type;

        size_type size;
        ar >> size;    //-V128

        t.clear();
        for (size_type i = 0; i < size; ++i)
        {
            value_type v;
            ar >> v;
            t.insert(t.end(), std::move(v));
        }
    }

    template <typename Key, typename Value, typename Hash, typename KeyEqual,
        typename Alloc>
    void serialize(output_archive& ar,
        const std::unordered_map<Key, Value, Hash, KeyEqual, Alloc>& t,
        unsigned)
    {
        using container_type =
            std::unordered_map<Key, Value, Hash, KeyEqual, Alloc>;

        using value_type = typename container_type::value_type;

        ar << t.size();    //-V128
        for (const value_type& val : t)
        {
            ar << val;
        }
    }
}}    // namespace hpx::serialization
