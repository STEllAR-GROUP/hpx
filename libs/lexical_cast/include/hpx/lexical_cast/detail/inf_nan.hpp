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

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

#include <hpx/lexical_cast/detail/lcast_char_constants.hpp>

namespace hpx { namespace util { namespace detail {

    inline bool lc_iequal(const char* val, const char* lcase, const char* ucase,
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
    template <class T>
    inline bool parse_inf_nan_impl(const char* begin, const char* end, T& value,
        const char* lc_NAN, const char* lc_nan, const char* lc_INFINITY,
        const char* lc_infinity, const char opening_brace,
        const char closing_brace) noexcept
    {
        using namespace std;
        if (begin == end)
            return false;
        const char minus = lcast_char_constants::minus;
        const char plus = lcast_char_constants::plus;
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
                value =
                    std::copysign(std::numeric_limits<T>::quiet_NaN(), T(-1));
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
                value =
                    std::copysign(std::numeric_limits<T>::infinity(), T(-1));
            return true;
        }

        return false;
    }

    template <class T>
    bool put_inf_nan_impl(char* begin, char*& end, const T& value,
        const char* lc_nan, const char* lc_infinity) noexcept
    {
        using namespace std;
        const char minus = lcast_char_constants::minus;
        if (std::isnan(value))
        {
            if (std::signbit(value))
            {
                *begin = minus;
                ++begin;
            }

            memcpy(begin, lc_nan, 3 * sizeof(char));
            end = begin + 3;
            return true;
        }
        else if (std::isinf(value))
        {
            if (std::signbit(value))
            {
                *begin = minus;
                ++begin;
            }

            memcpy(begin, lc_infinity, 3 * sizeof(char));
            end = begin + 3;
            return true;
        }

        return false;
    }

    template <class T>
    bool parse_inf_nan(const char* begin, const char* end, T& value) noexcept
    {
        return parse_inf_nan_impl(
            begin, end, value, "NAN", "nan", "INFINITY", "infinity", '(', ')');
    }

    template <class T>
    bool put_inf_nan(char* begin, char*& end, const T& value) noexcept
    {
        return put_inf_nan_impl(begin, end, value, "nan", "infinity");
    }

}}}    // namespace hpx::util::detail

#endif    // HPX_LEXICAL_CAST_DETAIL_INF_NAN_HPP
