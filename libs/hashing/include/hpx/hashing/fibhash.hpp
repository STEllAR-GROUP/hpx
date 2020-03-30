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
#include <cstdlib>

namespace hpx { namespace util {

    namespace detail {

        template <std::uint64_t N>
        struct hash_helper;

        template <>
        struct hash_helper<0>
        {
            HPX_STATIC_CONSTEXPR int log2 = -1;
        };

        template <std::uint64_t N>
        struct hash_helper
        {
            HPX_STATIC_CONSTEXPR std::uint64_t log2 =
                hash_helper<(N >> 1)>::log2 + 1;
            HPX_STATIC_CONSTEXPR std::uint64_t shift_amount = 64 - log2;
        };

        HPX_STATIC_CONSTEXPR std::uint64_t golden_ratio =
            11400714819323198485llu;
    }    // namespace detail

    // This function calculates the hash based on a multiplicative Fibonacci
    // scheme
    template <std::uint64_t N>
    constexpr std::uint64_t fibhash(std::uint64_t i)
    {
        using helper = detail::hash_helper<N>;
        static_assert(N != 0, "This algorithm only works with N != 0");
        static_assert(
            (1 << helper::log2) == N, "N must be a power of two");    // -V104
        return (detail::golden_ratio * (i ^ (i >> helper::shift_amount))) >>
            helper::shift_amount;
    }
}}    // namespace hpx::util
