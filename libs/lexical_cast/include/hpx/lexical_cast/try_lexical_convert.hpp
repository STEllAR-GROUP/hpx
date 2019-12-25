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

#ifndef HPX_LEXICAL_CAST_TRY_LEXICAL_CONVERT_HPP
#define HPX_LEXICAL_CAST_TRY_LEXICAL_CONVERT_HPP

#include <hpx/config.hpp>

#if defined(__clang__) ||                                                      \
    (defined(__GNUC__) &&                                                      \
        !(defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) ||     \
            defined(__ECC)) &&                                                 \
        (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#include <cstddef>
#include <string>
#include <type_traits>

#include <hpx/lexical_cast/detail/converter_lexical.hpp>
#include <hpx/lexical_cast/detail/converter_numeric.hpp>
#include <hpx/lexical_cast/detail/cstring_wrapper.hpp>
#include <hpx/lexical_cast/detail/is_character.hpp>

namespace hpx { namespace util {
    namespace detail {
        template <typename T>
        struct is_stdstring : std::false_type
        {
        };

        template <typename CharT, typename Traits, typename Alloc>
        struct is_stdstring<std::basic_string<CharT, Traits, Alloc>>
          : std::true_type
        {
        };

        template <typename Target, typename Source>
        struct is_arithmetic_and_not_xchars
          : std::integral_constant<bool,
                !(detail::is_character<Target>::value) &&
                    !(detail::is_character<Source>::value) &&
                    std::is_arithmetic<Source>::value &&
                    std::is_arithmetic<Target>::value>
        {
        };

        /*
         * is_char_to_char<Target, Source>::value is true,
         * Target and Souce are char types.
         */
        template <typename Target, typename Source>
        struct is_char_to_char
          : std::integral_constant<bool,
                detail::is_character<Target>::value &&
                    detail::is_character<Source>::value>

        {
        };

        template <typename Target, typename Source>
        struct is_char_array_to_stdstring : std::false_type
        {
        };

        template <typename CharT, typename Traits, typename Alloc>
        struct is_char_array_to_stdstring<
            std::basic_string<CharT, Traits, Alloc>, char*> : std::true_type
        {
        };

        template <typename CharT, typename Traits, typename Alloc>
        struct is_char_array_to_stdstring<
            std::basic_string<CharT, Traits, Alloc>, const char*>
          : std::true_type
        {
        };

        template <typename Target, typename Source>
        struct copy_converter_impl
        {
            template <class T>
            static inline bool try_convert(T&& arg, Target& result)
            {
                // equal to `result = std::forward<T>(arg);`
                result = static_cast<T&&>(arg);
                return true;
            }
        };

        template <typename Target, typename Source>
        inline bool try_lexical_convert(const Source& arg, Target& result)
        {
            typedef typename detail::array_to_pointer_decay<Source>::type src;

            typedef std::integral_constant<bool,
                detail::is_char_to_char<Target, src>::value ||
                    detail::is_char_array_to_stdstring<Target, src>::value ||
                    (std::is_same<Target, src>::value &&
                        detail::is_stdstring<Target>::value) ||
                    (std::is_same<Target, src>::value &&
                        detail::is_character<Target>::value)>
                shall_we_copy_t;

            typedef detail::is_arithmetic_and_not_xchars<Target, src>
                shall_we_copy_with_dynamic_check_t;

            // We do evaluate second `if_` lazily to avoid unnecessary instantiations
            // of `shall_we_copy_with_dynamic_check_t` and improve compilation times.
            typedef typename std::conditional<shall_we_copy_t::value,
                std::decay<detail::copy_converter_impl<Target, src>>,
                std::conditional<shall_we_copy_with_dynamic_check_t::value,
                    detail::dynamic_num_converter_impl<Target, src>,
                    detail::lexical_converter_impl<Target, src>>>::type
                caster_type_lazy;

            typedef typename caster_type_lazy::type caster_type;

            return caster_type::try_convert(arg, result);
        }

        template <typename Target>
        inline bool try_lexical_convert(
            const char* chars, std::size_t count, Target& result)
        {
            return detail::try_lexical_convert(
                detail::cstring_wrapper(chars, count), result);
        }

    }    // namespace detail

    // ADL barrier
    using ::hpx::util::detail::try_lexical_convert;

}}    // namespace hpx::util

#if defined(__clang__) ||                                                      \
    (defined(__GNUC__) &&                                                      \
        !(defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) ||     \
            defined(__ECC)) &&                                                 \
        (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)))
#pragma GCC diagnostic pop
#endif

#endif    // HPX_LEXICAL_CAST_TRY_LEXICAL_CONVERT_HPP
