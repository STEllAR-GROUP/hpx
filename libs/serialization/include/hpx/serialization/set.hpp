//  Copyright (c) 2015 Andreas Schaefer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/serialization_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>

namespace hpx { namespace serialization {

    template <typename T, typename Compare, typename Allocator>
    void serialize(
        input_archive& ar, std::set<T, Compare, Allocator>& set, unsigned)
    {
        set.clear();
        std::uint64_t size;
        ar >> size;

        set.clear();
        for (std::size_t i = 0; i < size; ++i)
        {
            T t;
            ar >> t;
            set.insert(set.end(), std::move(t));
        }
    }

    template <typename T, typename Compare, typename Allocator>
    void serialize(output_archive& ar,
        std::set<T, Compare, Allocator> const& set, unsigned)
    {
        std::uint64_t size = set.size();
        ar << size;
        if (set.empty())
            return;
        for (T const& i : set)
        {
            ar << i;
        }
    }
}}    // namespace hpx::serialization
