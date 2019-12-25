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

#ifndef HPX_LEXICAL_CAST_DETAIL_LCAST_UNSIGNED_CONVERTERS_HPP
#define HPX_LEXICAL_CAST_DETAIL_LCAST_UNSIGNED_CONVERTERS_HPP

#include <hpx/config.hpp>

#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

#include <hpx/lexical_cast/detail/lcast_char_constants.hpp>

namespace hpx { namespace util { namespace detail {

    template <class T>
    inline typename std::make_unsigned<T>::type lcast_to_unsigned(
        const T value) noexcept
    {
        typedef typename std::make_unsigned<T>::type result_type;
        return value < 0 ?
            static_cast<result_type>(0u - static_cast<result_type>(value)) :
            static_cast<result_type>(value);
    }

    template <class Traits, class T, class CharT>
    class lcast_put_unsigned
    {
        HPX_NON_COPYABLE(lcast_put_unsigned);

        typedef typename Traits::int_type int_type;
        typename std::conditional<(sizeof(unsigned) > sizeof(T)), unsigned,
            T>::type m_value;
        CharT* m_finish;
        CharT const m_czero;
        int_type const m_zero;

    public:
        lcast_put_unsigned(const T n_param, CharT* finish) noexcept
          : m_value(n_param)
          , m_finish(finish)
          , m_czero(lcast_char_constants<CharT>::zero)
          , m_zero(Traits::to_int_type(m_czero))
        {
            static_assert(!std::numeric_limits<T>::is_signed, "");
        }

        CharT* convert()
        {
            return main_convert_loop();
        }

    private:
        inline bool main_convert_iteration() noexcept
        {
            --m_finish;
            int_type const digit = static_cast<int_type>(m_value % 10U);
            Traits::assign(*m_finish, Traits::to_char_type(m_zero + digit));
            m_value /= 10;
            return !!m_value;    // suppressing warnings
        }

        inline CharT* main_convert_loop() noexcept
        {
            while (main_convert_iteration())
                ;
            return m_finish;
        }
    };

    template <class Traits, class T, class CharT>
    class lcast_ret_unsigned
    {
        HPX_NON_COPYABLE(lcast_ret_unsigned);

        bool m_multiplier_overflowed;
        T m_multiplier;
        T& m_value;
        const CharT* const m_begin;
        const CharT* m_end;

    public:
        lcast_ret_unsigned(
            T& value, const CharT* const begin, const CharT* end) noexcept
          : m_multiplier_overflowed(false)
          , m_multiplier(1)
          , m_value(value)
          , m_begin(begin)
          , m_end(end)
        {
            static_assert(!std::numeric_limits<T>::is_signed, "");
        }

        inline bool convert()
        {
            CharT const czero = lcast_char_constants<CharT>::zero;
            --m_end;
            m_value = static_cast<T>(0);

            if (m_begin > m_end || *m_end < czero || *m_end >= czero + 10)
                return false;
            m_value = static_cast<T>(*m_end - czero);
            --m_end;

            return main_convert_loop();
        }

    private:
        // Iteration that does not care about grouping/separators and assumes that all
        // input characters are digits
        inline bool main_convert_iteration() noexcept
        {
            CharT const czero = lcast_char_constants<CharT>::zero;
            T const maxv = (std::numeric_limits<T>::max)();

            m_multiplier_overflowed =
                m_multiplier_overflowed || (maxv / 10 < m_multiplier);
            m_multiplier = static_cast<T>(m_multiplier * 10);

            T const dig_value = static_cast<T>(*m_end - czero);
            T const new_sub_value = static_cast<T>(m_multiplier * dig_value);

            // We must correctly handle situations like `000000000000000000000000000001`.
            // So we take care of overflow only if `dig_value` is not '0'.
            if (*m_end < czero ||
                *m_end >= czero + 10    // checking for correct digit
                || (dig_value &&
                       (    // checking for overflow of ...
                           m_multiplier_overflowed    // ... multiplier
                           || static_cast<T>(maxv / dig_value) <
                               m_multiplier    // ... subvalue
                           || static_cast<T>(maxv - new_sub_value) <
                               m_value    // ... whole expression
                           )))
                return false;

            m_value = static_cast<T>(m_value + new_sub_value);

            return true;
        }

        bool main_convert_loop() noexcept
        {
            for (; m_end >= m_begin; --m_end)
            {
                if (!main_convert_iteration())
                {
                    return false;
                }
            }

            return true;
        }
    };

}}}    // namespace hpx::util::detail

#endif    // HPX_LEXICAL_CAST_DETAIL_LCAST_UNSIGNED_CONVERTERS_HPP
