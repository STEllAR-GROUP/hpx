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

#ifndef HPX_LEXICAL_CAST_DETAIL_CONVERTER_LEXICAL_HPP
#define HPX_LEXICAL_CAST_DETAIL_CONVERTER_LEXICAL_HPP

#include <hpx/config.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include <hpx/lexical_cast/detail/converter_lexical_streams.hpp>
#include <hpx/lexical_cast/detail/cstring_wrapper.hpp>
#include <hpx/lexical_cast/detail/is_character.hpp>
#include <hpx/lexical_cast/detail/lcast_precision.hpp>

namespace hpx { namespace util { namespace detail {

    // We are attempting to get char_traits<> from T
    // template parameter. Otherwise we'll be using std::char_traits<Char>
    template <class Char, class T>
    struct extract_char_traits : std::false_type
    {
        typedef std::char_traits<Char> trait_t;
    };

    template <class Char, class Traits, class Alloc>
    struct extract_char_traits<Char, std::basic_string<Char, Traits, Alloc>>
      : std::true_type
    {
        typedef Traits trait_t;
    };

    template <class T>
    struct array_to_pointer_decay
    {
        typedef T type;
    };

    template <class T, std::size_t N>
    struct array_to_pointer_decay<T[N]>
    {
        typedef const T* type;
    };

    // Return max. length of string representation of Source;
    template <class Source,    // Source type of lexical_cast.
        class Enable = void    // helper type
        >
    struct lcast_src_length
    {
        HPX_STATIC_CONSTEXPR std::size_t value = 1;
    };

    // Helper for integral types.
    // Notes on length calculation:
    // Max length for 32bit int with grouping "\1" and thousands_sep ',':
    // "-2,1,4,7,4,8,3,6,4,7"
    //  ^                    - is_signed
    //   ^                   - 1 digit not counted by digits10
    //    ^^^^^^^^^^^^^^^^^^ - digits10 * 2
    //
    // Constant is_specialized is used instead of constant 1
    // to prevent buffer overflow in a rare case when
    // <boost/limits.hpp> doesn't add missing specialization for
    // numeric_limits<T> for some integral type T.
    // When is_specialized is false, the whole expression is 0.
    template <class Source>
    struct lcast_src_length<Source,
        typename std::enable_if<std::is_integral<Source>::value>::type>
    {
        HPX_STATIC_CONSTEXPR std::size_t value =
            std::numeric_limits<Source>::is_signed +
            std::numeric_limits<Source>::is_specialized + /* == 1 */
            std::numeric_limits<Source>::digits10 * 2;
    };

    // Helper for floating point types.
    // -1.23456789e-123456
    // ^                   sign
    //  ^                  leading digit
    //   ^                 decimal point
    //    ^^^^^^^^         lcast_precision<Source>::value
    //            ^        "e"
    //             ^       exponent sign
    //              ^^^^^^ exponent (assumed 6 or less digits)
    // sign + leading digit + decimal point + "e" + exponent sign == 5
    template <class Source>
    struct lcast_src_length<Source,
        typename std::enable_if<std::is_floating_point<Source>::value>::type>
    {
        static_assert(std::numeric_limits<Source>::max_exponent10 <= 999999L &&
                std::numeric_limits<Source>::min_exponent10 >= -999999L,
            "Floating point out of range.");

        HPX_STATIC_CONSTEXPR std::size_t value =
            5 + lcast_precision<Source>::value + 6;
    };

    template <class Source, class Target>
    struct lexical_cast_stream_traits
    {
        typedef typename std::decay<Source>::type decayed_src;

        typedef typename std::conditional<
            detail::extract_char_traits<char, Target>::value,
            detail::extract_char_traits<char, Target>,
            detail::extract_char_traits<char, decayed_src>>::type::trait_t
            traits;

        typedef detail::lcast_src_length<decayed_src> len_t;
    };

    template <typename Target, typename Source>
    struct lexical_converter_impl
    {
        typedef lexical_cast_stream_traits<Source, Target> stream_trait;

        typedef detail::lexical_istream_limited_src<
            typename stream_trait::traits, stream_trait::len_t::value + 1>
            i_interpreter_type;

        typedef detail::lexical_ostream_limited_src<
            typename stream_trait::traits>
            o_interpreter_type;

        static inline bool try_convert(const Source& arg, Target& result)
        {
            i_interpreter_type i_interpreter;

            // Disabling ADL, by directly specifying operators.
            if (!(i_interpreter.operator<<(arg)))
                return false;

            o_interpreter_type out(
                i_interpreter.cbegin(), i_interpreter.cend());

            // Disabling ADL, by directly specifying operators.
            if (!(out.operator>>(result)))
                return false;

            return true;
        }
    };

}}}    // namespace hpx::util::detail

#endif    // HPX_LEXICAL_CAST_DETAIL_CONVERTER_LEXICAL_HPP
