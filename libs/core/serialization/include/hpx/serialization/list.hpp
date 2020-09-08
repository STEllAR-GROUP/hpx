//  Copyright (c) 2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/detail/serialize_collection.hpp>
#include <hpx/serialization/serialize.hpp>

#include <cstdint>
#include <list>

namespace hpx { namespace serialization {

    template <typename T, typename Allocator>
    void serialize(input_archive& ar, std::list<T, Allocator>& ls, unsigned)
    {
        // normal load ...
        std::uint64_t size;
        ar >> size;    //-V128
        if (size == 0)
            return;

        detail::load_collection(ar, ls, size);
    }

    template <typename T, typename Allocator>
    void serialize(
        output_archive& ar, const std::list<T, Allocator>& ls, unsigned)
    {
        // normal save ...
        std::uint64_t size = ls.size();
        ar << size;
        if (ls.empty())
            return;

        detail::save_collection(ar, ls);
    }
}}    // namespace hpx::serialization
