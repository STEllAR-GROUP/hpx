//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_STRING_HPP
#define HPX_SERIALIZATION_STRING_HPP

#include <hpx/runtime/serialization/serialize.hpp>

#include <string>

namespace hpx { namespace serialization
{
    // load string
    template <typename Char, typename CharTraits, typename Allocator>
    void serialize(input_archive & ar, std::basic_string<Char, CharTraits,
        Allocator> & s, unsigned)
    {
        typedef std::basic_string<Char, CharTraits, Allocator> string_type;
        typedef typename string_type::size_type size_type;
        size_type size = 0;
        ar >> size; //-V128

        s.clear();
        s.resize(size);

        load_binary(ar, &s[0], size * sizeof(Char));
    }

    // save string
    template <typename Char, typename CharTraits, typename Allocator>
    void serialize(output_archive & ar, std::basic_string<Char, CharTraits,
        Allocator> & s, unsigned)
    {
        ar << s.size(); //-V128
        save_binary(ar, s.data(), s.size() * sizeof(Char));
    }
}}

#endif
