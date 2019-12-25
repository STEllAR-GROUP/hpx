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

#ifndef HPX_LEXICAL_CAST_DETAIL_INF_NAN_HPP
#define HPX_LEXICAL_CAST_DETAIL_INF_NAN_HPP

#include <hpx/config.hpp>

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/sign.hpp>
#include <cstddef>
#include <cstring>
#include <limits>

#include <hpx/lexical_cast/detail/lcast_char_constants.hpp>

namespace hpx { namespace util { namespace detail {
    template <class CharT>
    bool lc_iequal(const CharT* val, const CharT* lcase, const CharT* ucase,
        unsigned int len) noexcept
    {
        for (unsigned int i = 0; i < len; ++i)
        {
            if (val[i] != lcase[i] && val[i] != ucase[i])
                return false;
        }

        return true;
    }

    /* Returns true and sets the correct value if found NaN or Inf. */
    template <class CharT, class T>
    inline bool parse_inf_nan_impl(const CharT* begin, const CharT* end,
        T& value, const CharT* lc_NAN, const CharT* lc_nan,
        const CharT* lc_INFINITY, const CharT* lc_infinity,
        const CharT opening_brace, const CharT closing_brace) noexcept
    {
        using namespace std;
        if (begin == end)
            return false;
        const CharT minus = lcast_char_constants<CharT>::minus;
        const CharT plus = lcast_char_constants<CharT>::plus;
        const int inifinity_size = 8;    // == sizeof("infinity") - 1

        /* Parsing +/- */
        bool const has_minus = (*begin == minus);
        if (has_minus || *begin == plus)
        {
            ++begin;
        }

        if (end - begin < 3)
            return false;
        if (lc_iequal(begin, lc_nan, lc_NAN, 3))
        {
            begin += 3;
            if (end != begin)
            {
                /* It is 'nan(...)' or some bad input*/

                if (end - begin < 2)
                    return false;    // bad input
                --end;
                if (*begin != opening_brace || *end != closing_brace)
                    return false;    // bad input
            }

            if (!has_minus)
                value = std::numeric_limits<T>::quiet_NaN();
            else
                value = (boost::math::changesign)(
                    std::numeric_limits<T>::quiet_NaN());
            return true;
        }
        else if ((                       /* 'INF' or 'inf' */
                     end - begin == 3    // 3 == sizeof('inf') - 1
                     && lc_iequal(begin, lc_infinity, lc_INFINITY, 3)) ||
            (/* 'INFINITY' or 'infinity' */
                end - begin == inifinity_size &&
                lc_iequal(begin, lc_infinity, lc_INFINITY, inifinity_size)))
        {
            if (!has_minus)
                value = std::numeric_limits<T>::infinity();
            else
                value = (boost::math::changesign)(
                    std::numeric_limits<T>::infinity());
            return true;
        }

        return false;
    }

    template <class CharT, class T>
    bool put_inf_nan_impl(CharT* begin, CharT*& end, const T& value,
        const CharT* lc_nan, const CharT* lc_infinity) noexcept
    {
        using namespace std;
        const CharT minus = lcast_char_constants<CharT>::minus;
        if ((boost::math::isnan)(value))
        {
            if ((boost::math::signbit)(value))
            {
                *begin = minus;
                ++begin;
            }

            memcpy(begin, lc_nan, 3 * sizeof(CharT));
            end = begin + 3;
            return true;
        }
        else if ((boost::math::isinf)(value))
        {
            if ((boost::math::signbit)(value))
            {
                *begin = minus;
                ++begin;
            }

            memcpy(begin, lc_infinity, 3 * sizeof(CharT));
            end = begin + 3;
            return true;
        }

        return false;
    }

    template <class CharT, class T>
    bool parse_inf_nan(
        const CharT* begin, const CharT* end, T& value) noexcept
    {
        return parse_inf_nan_impl(
            begin, end, value, "NAN", "nan", "INFINITY", "infinity", '(', ')');
    }

    template <class CharT, class T>
    bool put_inf_nan(CharT* begin, CharT*& end, const T& value) noexcept
    {
        return put_inf_nan_impl(begin, end, value, "nan", "infinity");
    }
}}}    // namespace hpx::util::detail

#endif    // HPX_LEXICAL_CAST_DETAIL_INF_NAN_HPP
