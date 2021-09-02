//  Copyright (c) 2021 Srinivas Yadav
//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>

#if defined(HPX_HAVE_CXX20_EXPERIMENTAL_SIMD)

#include <experimental/simd>

#include <cstddef>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename T, std::size_t N, typename Abi>
        struct vector_pack_type
        {
            typedef std::experimental::fixed_size_simd<T, N> type;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 0, Abi>
        {
            typedef typename std::conditional<std::is_void<Abi>::value,
                std::experimental::simd_abi::native<T>, Abi>::type abi_type;

            typedef std::experimental::simd<T, abi_type> type;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 1, Abi>
        {
            typedef std::experimental::simd<T,
                std::experimental::simd_abi::scalar>
                type;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type : detail::vector_pack_type<T, N, Abi>
    {
    };
}}}    // namespace hpx::parallel::traits

#endif
