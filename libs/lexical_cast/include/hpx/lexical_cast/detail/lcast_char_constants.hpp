// Copyright Kevlin Henney, 2000-2005.
// Copyright Alexander Nasonov, 2006-2010.
// Copyright Antony Polukhin, 2011-2019.
// Copyright Agustin Berge, 2019.
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// what:  lexical_cast custom keyword cast
// who:   contributed by Kevlin Henney,
//        enhanced with contributions from Terje Slettebo,
//        with additional fixes and suggestions from Gennaro Prota,
//        Beman Dawes, Dave Abrahams, Daryle Walker, Peter Dimov,
//        Alexander Nasonov, Antony Polukhin, Justin Viiret, Michael Hofmann,
//        Cheng Yang, Matthew Bradbury, David W. Birdsall, Pavel Korzh and other Boosters
// when:  November 2000, March 2003, June 2005, June 2006, March 2011 - 2014

#ifndef HPX_LEXICAL_CAST_DETAIL_LCAST_CHAR_CONSTANTS_HPP
#define HPX_LEXICAL_CAST_DETAIL_LCAST_CHAR_CONSTANTS_HPP

#include <hpx/config.hpp>

namespace hpx { namespace util { namespace detail {

    // '0', '-', '+', 'e', 'E' and '.' constants
    template <typename Char>
    struct lcast_char_constants
    {
        // We check in tests assumption that static casted character is
        // equal to correctly written C++ literal: U'0' == static_cast<char32_t>('0')
        HPX_STATIC_CONSTEXPR Char zero = static_cast<Char>('0');
        HPX_STATIC_CONSTEXPR Char minus = static_cast<Char>('-');
        HPX_STATIC_CONSTEXPR Char plus = static_cast<Char>('+');
        HPX_STATIC_CONSTEXPR Char lowercase_e = static_cast<Char>('e');
        HPX_STATIC_CONSTEXPR Char capital_e = static_cast<Char>('E');
        HPX_STATIC_CONSTEXPR Char c_decimal_separator = static_cast<Char>('.');
    };

}}}    // namespace hpx::util::detail

#endif    // HPX_LEXICAL_CAST_DETAIL_LCAST_CHAR_CONSTANTS_HPP
