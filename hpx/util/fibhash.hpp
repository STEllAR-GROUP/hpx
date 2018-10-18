// Copyright (c) 2018 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This code is based on the article found here:
// https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/

#ifndef HPX_UTIL_FIBHASH_HPP
#define HPX_UTIL_FIBHASH_HPP

#include <cstdlib>
#include <cstddef>

namespace hpx { namespace util {
    namespace detail
    {
        template <std::size_t N>
        struct hash_helper;

        template <>
        struct hash_helper<0>
        {
            static constexpr int log2 = -1;
        };

        template <std::size_t N>
        struct hash_helper
        {
            static constexpr std::size_t log2 = hash_helper<(N >> 1)>::log2 + 1;
            static constexpr std::size_t shift_amount = 64 - log2;
        };

        constexpr std::size_t golden_ratio = 11400714819323198485llu;
    }

    // This function calculates the hash based on a multiplicative fibonacci
    // scheme
    template <std::size_t N>
    constexpr std::size_t fibhash(std::size_t i)
    {
        using helper = detail::hash_helper<N>;
        static_assert(N != 0, "This algorithm only works with N != 0");
        static_assert((1 << helper::log2) == N, "N must be a power of two");
        return
            (detail::golden_ratio * (i ^ (i >> helper::shift_amount)))
            >> helper::shift_amount;
    }
}}

#endif
