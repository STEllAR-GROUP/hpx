//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_STRING_HPP
#define HPX_SERIALIZATION_STRING_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/basic_archive.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>

#include <cstdint>
#include <string>

namespace hpx { namespace serialization
{
    // load string
    template <typename Char, typename CharTraits, typename Allocator>
    void serialize(input_archive & ar, std::basic_string<Char, CharTraits,
        Allocator> & s, unsigned)
    {
        std::uint64_t size = 0;
        ar >> size; //-V128

        s.clear();
        if (s.size() < size)
            s.resize(size);

        load_binary(ar, &s[0], size * sizeof(Char));
    }

    // save string
    template <typename Char, typename CharTraits, typename Allocator>
    void serialize(output_archive & ar, const std::basic_string<Char, CharTraits,
        Allocator> & s, unsigned)
    {
        std::uint64_t size = s.size();
        ar << size;
        save_binary(ar, s.data(), s.size() * sizeof(Char));
    }
}}

#endif
