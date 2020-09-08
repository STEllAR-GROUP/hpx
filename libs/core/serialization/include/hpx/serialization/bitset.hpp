//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/serialization_fwd.hpp>

#include <bitset>
#include <cstddef>
#include <string>

namespace hpx { namespace serialization {

    template <std::size_t N>
    void serialize(input_archive& ar, std::bitset<N>& d, unsigned)
    {
        std::string bits;
        ar >> bits;
        d = std::bitset<N>(bits);
    }

    template <std::size_t N>
    void serialize(output_archive& ar, std::bitset<N> const& bs, unsigned)
    {
        std::string const bits = bs.to_string();
        ar << bits;
    }
}}    // namespace hpx::serialization
