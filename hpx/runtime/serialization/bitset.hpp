//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_BITSET_HPP
#define HPX_SERIALIZATION_BITSET_HPP

#include <hpx/runtime/serialization/serialize.hpp>

#include <bitset>
#include <cstddef>
#include <string>

namespace hpx { namespace serialization
{
    template <std::size_t N>
    void serialize(input_archive & ar, std::bitset<N> & d, unsigned)
    {
        std::string bits;
        ar >> bits;
        d = std::bitset<N>(bits);
    }

    template <std::size_t N>
    void serialize(output_archive & ar, const std::bitset<N> & bs, unsigned)
    {
        std::string const bits = bs.to_string();
        ar << bits;
    }
}}

#endif
