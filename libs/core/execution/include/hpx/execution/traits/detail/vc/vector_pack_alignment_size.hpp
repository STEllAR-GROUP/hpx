//  Copyright (c) 2016-2022 Hartmut Kaiser
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
    template <typename T, typename Abi>
    struct is_vector_pack<Vc::Vector<T, Abi>> : std::true_type
    {
    };

    template <typename T, std::size_t N, typename V, std::size_t W>
    struct is_vector_pack<Vc::SimdArray<T, N, V, W>> : std::true_type
    {
    };

    template <typename T>
    struct is_vector_pack<Vc::Scalar::Vector<T>> : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    struct is_scalar_vector_pack<Vc::Vector<T, Abi>> : std::false_type
    {
    };

    template <typename T, std::size_t N, typename V, std::size_t W>
    struct is_scalar_vector_pack<Vc::SimdArray<T, N, V, W>>
      : std::integral_constant<bool, N == 1>
    {
    };

    template <typename T>
    struct is_scalar_vector_pack<Vc::Scalar::Vector<T>> : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    struct is_non_scalar_vector_pack<Vc::Vector<T, Abi>> : std::true_type
    {
    };

    template <typename T, std::size_t N, typename V, std::size_t W>
    struct is_non_scalar_vector_pack<Vc::SimdArray<T, N, V, W>>
      : std::integral_constant<bool, N != 1>
    {
    };

    template <typename T>
    struct is_non_scalar_vector_pack<Vc::Scalar::Vector<T>> : std::false_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable>
    struct vector_pack_alignment
    {
        static constexpr std::size_t const value =
            Vc::Vector<T>::MemoryAlignment;
    };

    template <typename T, typename Abi>
    struct vector_pack_alignment<Vc::Vector<T, Abi>>
    {
        static constexpr std::size_t const value =
            Vc::Vector<T, Abi>::MemoryAlignment;
    };

    template <typename T, std::size_t N, typename V, std::size_t W>
    struct vector_pack_alignment<Vc::SimdArray<T, N, V, W>>
    {
        static constexpr std::size_t const value =
            Vc::SimdArray<T, N, V, W>::MemoryAlignment;
    };

    template <typename T>
    struct vector_pack_alignment<Vc::Scalar::Vector<T>>
    {
        static constexpr std::size_t const value =
            Vc::Scalar::Vector<T>::MemoryAlignment;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable>
    struct vector_pack_size
    {
        static std::size_t const value = Vc::Vector<T>::Size;
    };

    template <typename T, typename Abi>
    struct vector_pack_size<Vc::Vector<T, Abi>>
    {
        static std::size_t const value = Vc::Vector<T, Abi>::Size;
    };

    template <typename T, std::size_t N, typename V, std::size_t W>
    struct vector_pack_size<Vc::SimdArray<T, N, V, W>>
    {
        static std::size_t const value = Vc::SimdArray<T, N, V, W>::Size;
    };

    template <typename T>
    struct vector_pack_size<Vc::Scalar::Vector<T>>
    {
        static std::size_t const value = Vc::Scalar::Vector<T>::Size;
    };
}    // namespace hpx::parallel::traits

#endif
