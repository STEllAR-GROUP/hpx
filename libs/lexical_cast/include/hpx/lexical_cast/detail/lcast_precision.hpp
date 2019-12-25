// Copyright Alexander Nasonov & Paul A. Bristow 2006.
// Copyright Agustin Berge, 2019.
//
// SPDX-License-Identifier: BSL-1.0
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LEXICAL_CAST_DETAIL_LCAST_PRECISION_HPP_INCLUDED
#define HPX_LEXICAL_CAST_DETAIL_LCAST_PRECISION_HPP_INCLUDED

#include <hpx/config.hpp>

#include <climits>
#include <ios>
#include <limits>
#include <type_traits>

namespace hpx { namespace util { namespace detail {

    class lcast_abstract_stub
    {
    };

    // Calculate an argument to pass to std::ios_base::precision from
    // lexical_cast. See alternative implementation for broken standard
    // libraries in lcast_get_precision below. Keep them in sync, please.
    template <class T>
    struct lcast_precision
    {
        typedef typename std::conditional<std::is_abstract<T>::value,
            std::numeric_limits<lcast_abstract_stub>,
            std::numeric_limits<T>>::type limits;

        HPX_STATIC_CONSTEXPR bool use_default_precision =
            !limits::is_specialized || limits::is_exact;

        HPX_STATIC_CONSTEXPR bool is_specialized_bin =
            !use_default_precision && limits::radix == 2 && limits::digits > 0;

        HPX_STATIC_CONSTEXPR bool is_specialized_dec = !use_default_precision &&
            limits::radix == 10 && limits::digits10 > 0;

        HPX_STATIC_CONSTEXPR std::streamsize streamsize_max =
            (std::numeric_limits<std::streamsize>::max)();

        HPX_STATIC_CONSTEXPR unsigned int precision_dec = limits::digits10 + 1U;

        static_assert(
            !is_specialized_dec || precision_dec <= streamsize_max + 0UL,
            "!is_specialized_dec || precision_dec <= streamsize_max + 0UL");

        HPX_STATIC_CONSTEXPR unsigned long precision_bin =
            2UL + limits::digits * 30103UL / 100000UL;

        static_assert(!is_specialized_bin ||
                (limits::digits + 0UL < ULONG_MAX / 30103UL &&
                    precision_bin > limits::digits10 + 0UL &&
                    precision_bin <= streamsize_max + 0UL),
            "!is_specialized_bin || "
            "(limits::digits + 0UL < ULONG_MAX / 30103UL && "
            "precision_bin > limits::digits10 + 0UL && "
            "precision_bin <= streamsize_max + 0UL)");

        HPX_STATIC_CONSTEXPR std::streamsize value = is_specialized_bin ?
            precision_bin :
            is_specialized_dec ? precision_dec : 6;
    };

    template <class T>
    inline std::streamsize lcast_get_precision(T* = nullptr)
    {
        return lcast_precision<T>::value;
    }

    template <class T>
    inline void lcast_set_precision(std::ios_base& stream, T*)
    {
        stream.precision(lcast_get_precision<T>());
    }

    template <class Source, class Target>
    inline void lcast_set_precision(std::ios_base& stream, Source*, Target*)
    {
        std::streamsize const s = lcast_get_precision(static_cast<Source*>(0));
        std::streamsize const t = lcast_get_precision(static_cast<Target*>(0));
        stream.precision(s > t ? s : t);
    }

}}}    // namespace hpx::util::detail

#endif    //  HPX_LEXICAL_CAST_DETAIL_LCAST_PRECISION_HPP_INCLUDED
