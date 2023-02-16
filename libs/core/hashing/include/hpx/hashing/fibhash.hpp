// Copyright (c) 2018 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This code is based on the article found here:
// https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

namespace hpx::util {

    namespace detail {

        template <std::uint64_t N>
        inline constexpr std::uint64_t log2 = log2<(N >> 1)> + 1;    //-V573

        template <>
        inline constexpr std::uint64_t log2<0> = -1;

        template <std::uint64_t N>
        inline constexpr std::uint64_t shift_amount = 64 - log2<N>;

        inline constexpr std::uint64_t golden_ratio = 11400714819323198485llu;
    }    // namespace detail

    // This function calculates the hash based on a multiplicative Fibonacci
    // scheme
    template <std::uint64_t N>
    constexpr std::uint64_t fibhash(std::uint64_t i) noexcept
    {
        static_assert(N != 0, "This algorithm only works with N != 0");
        static_assert((1 << detail::log2<N>) == N,
            "N must be a power of two");    // -V104

        return (detail::golden_ratio * (i ^ (i >> detail::shift_amount<N>) )) >>
            detail::shift_amount<N>;
    }
}    // namespace hpx::util
