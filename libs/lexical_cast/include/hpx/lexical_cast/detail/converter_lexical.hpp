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

#include <boost/detail/lcast_precision.hpp>
#include <boost/type_traits/has_left_shift.hpp>
#include <boost/type_traits/has_right_shift.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include <hpx/lexical_cast/detail/converter_lexical_streams.hpp>
#include <hpx/lexical_cast/detail/cstring_wrapper.hpp>
#include <hpx/lexical_cast/detail/is_character.hpp>

namespace hpx { namespace util { namespace detail {

    // Helper type, meaning that stram character for T must be deduced
    // at Stage 2 (See deduce_source_char<T> and deduce_target_char<T>)
    template <class T>
    struct deduce_character_type_later
    {
    };

    // Selectors to choose stream character type (common for Source and Target)
    // Returns one of char or deduce_character_type_later<T> types
    // Executed on Stage 1 (See deduce_source_char<T> and deduce_target_char<T>)
    template <typename Type>
    struct stream_char_common
      : public std::conditional<detail::is_character<Type>::value, Type,
            detail::deduce_character_type_later<Type>>
    {
    };

    template <typename Char>
    struct stream_char_common<Char*>
      : public std::conditional<detail::is_character<Char>::value, Char,
            detail::deduce_character_type_later<Char*>>
    {
    };

    template <typename Char>
    struct stream_char_common<const Char*>
      : public std::conditional<detail::is_character<Char>::value, Char,
            detail::deduce_character_type_later<const Char*>>
    {
    };

    template <typename Char>
    struct stream_char_common<detail::cstring_wrapper<Char>>
      : public std::conditional<detail::is_character<Char>::value, Char,
            detail::deduce_character_type_later<
                detail::cstring_wrapper<const Char*>>>
    {
    };

    template <typename Char>
    struct stream_char_common<detail::cstring_wrapper<const Char>>
      : public std::conditional<detail::is_character<Char>::value, Char,
            detail::deduce_character_type_later<
                detail::cstring_wrapper<const Char*>>>
    {
    };

    template <class Char, class Traits, class Alloc>
    struct stream_char_common<std::basic_string<Char, Traits, Alloc>>
    {
        typedef Char type;
    };

    template <typename Char, std::size_t N>
    struct stream_char_common<std::array<Char, N>>
      : public std::conditional<detail::is_character<Char>::value, Char,
            detail::deduce_character_type_later<std::array<Char, N>>>
    {
    };

    template <typename Char, std::size_t N>
    struct stream_char_common<std::array<const Char, N>>
      : public std::conditional<detail::is_character<Char>::value, Char,
            detail::deduce_character_type_later<std::array<const Char, N>>>
    {
    };

    // If type T is `deduce_character_type_later` type, then tries to deduce
    // character type using boost::has_left_shift<T> metafunction.
    // Otherwise supplied type T is a character type.
    // Executed at Stage 2  (See deduce_source_char<T> and deduce_target_char<T>)
    template <class Char>
    struct deduce_source_char_impl
    {
        typedef Char type;
    };

    template <class T>
    struct deduce_source_char_impl<deduce_character_type_later<T>>
    {
        typedef boost::has_left_shift<std::basic_ostream<char>, T> result_t;

        static_assert(result_t::value, "Source type is not std::ostream`able");
        typedef char type;
    };

    // If type T is `deduce_character_type_later` type, then tries to deduce
    // character type using boost::has_right_shift<T> metafunction.
    // Otherwise supplied type T is a character type.
    // Executed at Stage 2  (See deduce_source_char<T> and deduce_target_char<T>)
    template <class Char>
    struct deduce_target_char_impl
    {
        typedef Char type;
    };

    template <class T>
    struct deduce_target_char_impl<deduce_character_type_later<T>>
    {
        typedef boost::has_right_shift<std::basic_istream<char>, T> result_t;

        static_assert(result_t::value, "Target type is not std::istream`able");
        typedef char type;
    };

    // We deduce stream character types in two stages.
    //
    // Stage 1 is common for Target and Source. At Stage 1 we get
    // character type or deduce_character_type_later<T> where T is the
    // original type.
    // Stage 1 is executed by stream_char_common<T>
    //
    // At Stage 2 we normalize character types or try to deduce character
    // type using metafunctions.
    // Stage 2 is executed by deduce_target_char_impl<T> and
    // deduce_source_char_impl<T>
    //
    // deduce_target_char<T> and deduce_source_char<T> functions combine
    // both stages

    template <class T>
    struct deduce_target_char
    {
        typedef typename stream_char_common<T>::type stage1_type;
        typedef typename deduce_target_char_impl<stage1_type>::type stage2_type;

        typedef stage2_type type;
    };

    template <class T>
    struct deduce_source_char
    {
        typedef typename stream_char_common<T>::type stage1_type;
        typedef typename deduce_source_char_impl<stage1_type>::type stage2_type;

        typedef stage2_type type;
    };

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
        typedef typename detail::array_to_pointer_decay<Source>::type src;
        typedef typename std::remove_cv<src>::type no_cv_src;

        typedef detail::deduce_source_char<no_cv_src> deduce_src_char_metafunc;
        typedef typename deduce_src_char_metafunc::type src_char_t;
        typedef typename detail::deduce_target_char<Target>::type target_char_t;
        typedef target_char_t char_type;

        typedef typename std::conditional<
            detail::extract_char_traits<char_type, Target>::value,
            typename detail::extract_char_traits<char_type, Target>,
            typename detail::extract_char_traits<char_type,
                no_cv_src>>::type::trait_t traits;

        typedef std::integral_constant<bool,
            !(std::is_integral<no_cv_src>::value ||
                detail::is_character<typename deduce_src_char_metafunc::
                        stage1_type    // if we did not get character type at stage1
                    >::value    // then we have no optimization for that type
                )>
            is_source_input_not_optimized_t;

        // If we have an optimized conversion for
        // Source, we do not need to construct stringbuf.
        HPX_STATIC_CONSTEXPR bool requires_stringbuf =
            (is_source_input_not_optimized_t::value);

        typedef detail::lcast_src_length<no_cv_src> len_t;
    };

    template <typename Target, typename Source>
    struct lexical_converter_impl
    {
        typedef lexical_cast_stream_traits<Source, Target> stream_trait;

        typedef detail::lexical_istream_limited_src<
            typename stream_trait::char_type, typename stream_trait::traits,
            stream_trait::requires_stringbuf, stream_trait::len_t::value + 1>
            i_interpreter_type;

        typedef detail::lexical_ostream_limited_src<
            typename stream_trait::char_type, typename stream_trait::traits>
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
