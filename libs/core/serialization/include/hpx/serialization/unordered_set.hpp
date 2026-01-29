//  Copyright (c) 2015 Andreas Schaefer
//  Copyright (c) 2023-2026 Hartmut Kaiser
//  Copyright (c) 2026 Ujjwal Shekhar
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <utility>

namespace hpx::serialization {

    HPX_CXX_EXPORT template <typename T, typename Hash, typename KeyEqual,
        typename Allocator>
    void serialize(input_archive& ar,
        std::unordered_set<T, Hash, KeyEqual, Allocator>& set, unsigned)
    {
        std::uint64_t size;
        ar >> size;

        set.clear();
        for (std::size_t i = 0; i < size; ++i)
        {
            T t;
            ar >> t;
            set.insert(set.end(), HPX_MOVE(t));
        }
    }

    HPX_CXX_EXPORT template <typename T, typename Hash, typename KeyEqual,
        typename Allocator>
    void serialize(output_archive& ar,
        std::unordered_set<T, Hash, KeyEqual, Allocator> const& set, unsigned)
    {
        std::uint64_t const size = set.size();
        ar << size;
        if (size == 0)
            return;

        for (T const& val : set)
        {
            ar << val;
        }
    }

    HPX_CXX_EXPORT template <typename T, typename Hash, typename KeyEqual,
        typename Allocator>
    void serialize(input_archive& ar,
        std::unordered_multiset<T, Hash, KeyEqual, Allocator>& set, unsigned)
    {
        std::uint64_t size;
        ar >> size;

        set.clear();
        for (std::size_t i = 0; i < size; ++i)
        {
            T t;
            ar >> t;
            set.insert(set.end(), HPX_MOVE(t));
        }
    }

    HPX_CXX_EXPORT template <typename T, typename Hash, typename KeyEqual,
        typename Allocator>
    void serialize(output_archive& ar,
        std::unordered_multiset<T, Hash, KeyEqual, Allocator> const& set,
        unsigned)
    {
        std::uint64_t const size = set.size();
        ar << size;
        if (size == 0)
            return;

        for (T const& val : set)
        {
            ar << val;
        }
    }
}    // namespace hpx::serialization