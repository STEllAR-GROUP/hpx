//  Copyright (c) 2021 Srinivas Yadav
//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_EXPERIMENTAL_SIMD)

#include <hpx/execution/traits/detail/simd/vector_pack_simd.hpp>

#include <cstddef>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename T, std::size_t N, typename Abi>
        struct vector_pack_type
        {
            typedef datapar::experimental::fixed_size_simd<T, N> type;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 0, Abi>
        {
            typedef typename std::conditional<std::is_void<Abi>::value,
                datapar::experimental::native<T>, Abi>::type abi_type;

            typedef datapar::experimental::simd<T, abi_type> type;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 1, Abi>
        {
            using type = T;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type : detail::vector_pack_type<T, N, Abi>
    {
    };

    ////////////////////////////////////////////////////////////////////
    template <typename T>
    struct vector_pack_mask_type<T,
        typename std::enable_if_t<datapar::experimental::is_simd_v<T>>>
    {
        using type = T::mask_type;
    };
}}}    // namespace hpx::parallel::traits

#endif
