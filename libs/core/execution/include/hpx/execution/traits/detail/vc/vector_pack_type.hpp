//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)

#include <cstddef>
#include <type_traits>

#include <Vc/Vc>
#include <Vc/global.h>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::traits {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename T, std::size_t N, typename Abi>
        struct vector_pack_type
        {
            using type = Vc::SimdArray<T, N>;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 0, Abi>
        {
            using abi_type = std::conditional_t<std::is_void_v<Abi>,
                Vc::VectorAbi::Best<T>, Abi>;

            using type = Vc::Vector<T, abi_type>;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 1, Abi>
        {
            using type = Vc::Scalar::Vector<T>;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type : detail::vector_pack_type<T, N, Abi>
    {
    };

    // don't wrap types twice
    template <typename T, std::size_t N, typename Abi1, typename Abi2>
    struct vector_pack_type<Vc::Vector<T, Abi1>, N, Abi2>
    {
        using type = Vc::Vector<T, Abi1>;
    };

    template <typename T, std::size_t N1, typename V, std::size_t W,
        std::size_t N2, typename Abi>
    struct vector_pack_type<Vc::SimdArray<T, N1, V, W>, N2, Abi>
    {
        using type = Vc::SimdArray<T, N1, V, W>;
    };

    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type<Vc::Scalar::Vector<T>, N, Abi>
    {
        using type = Vc::Scalar::Vector<T>;
    };

    ////////////////////////////////////////////////////////////////////
    template <typename T>
    struct vector_pack_mask_type<T,
        typename std::enable_if_t<Vc::Traits::is_simd_vector<T>::value>>
    {
        using type = typename T::mask_type;
    };
}    // namespace hpx::parallel::traits

#endif
