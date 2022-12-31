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
#include <iterator>
#include <memory>

#include <Vc/Vc>
#include <Vc/global.h>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi, typename NewT>
    struct rebind_pack<Vc::Vector<T, Abi>, NewT>
    {
        using type = Vc::Vector<NewT, Abi>;
    };

    template <typename T, std::size_t N, typename V, std::size_t W,
        typename NewT>
    struct rebind_pack<Vc::SimdArray<T, N, V, W>, NewT>
    {
        using type = Vc::SimdArray<NewT, N, V, W>;
    };

    template <typename T, typename NewT>
    struct rebind_pack<Vc::Scalar::Vector<T>, NewT>
    {
        using type = Vc::Scalar::Vector<NewT>;
    };

    // don't wrap types twice
    template <typename T, typename Abi1, typename NewT, typename Abi2>
    struct rebind_pack<Vc::Vector<T, Abi1>, Vc::Vector<NewT, Abi2>>
    {
        using type = Vc::Vector<NewT, Abi2>;
    };

    template <typename T, std::size_t N1, typename V1, std::size_t W1,
        typename NewT, std::size_t N2, typename V2, std::size_t W2>
    struct rebind_pack<Vc::SimdArray<T, N1, V1, W1>,
        Vc::SimdArray<NewT, N2, V2, W2>>
    {
        using type = Vc::SimdArray<NewT, N2, V2, W2>;
    };

    template <typename T, typename NewT>
    struct rebind_pack<Vc::Scalar::Vector<T>, Vc::Scalar::Vector<NewT>>
    {
        using type = Vc::Scalar::Vector<NewT>;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_load
    {
        using value_type = typename rebind_pack<V, ValueType>::type;

        template <typename Iter>
        static value_type aligned(Iter const& iter)
        {
            return value_type(std::addressof(*iter), Vc::Aligned);
        }

        template <typename Iter>
        static value_type unaligned(Iter const& iter)
        {
            return value_type(std::addressof(*iter), Vc::Unaligned);
        }
    };

    template <typename V, typename T, typename Abi>
    struct vector_pack_load<V, Vc::Vector<T, Abi>>
    {
        using value_type = typename rebind_pack<V, Vc::Vector<T, Abi>>::type;

        template <typename Iter>
        static value_type aligned(Iter const& iter)
        {
            return *iter;
        }

        template <typename Iter>
        static value_type unaligned(Iter const& iter)
        {
            return *iter;
        }
    };

    template <typename Value, typename T, std::size_t N, typename V,
        std::size_t W>
    struct vector_pack_load<Value, Vc::SimdArray<T, N, V, W>>
    {
        using value_type =
            typename rebind_pack<Value, Vc::SimdArray<T, N, V, W>>::type;

        template <typename Iter>
        static value_type aligned(Iter const& iter)
        {
            return *iter;
        }

        template <typename Iter>
        static value_type unaligned(Iter const& iter)
        {
            return *iter;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_store
    {
        template <typename Iter>
        static void aligned(V const& value, Iter const& iter)
        {
            value.store(std::addressof(*iter), Vc::Aligned);
        }

        template <typename Iter>
        static void unaligned(V const& value, Iter const& iter)
        {
            value.store(std::addressof(*iter), Vc::Unaligned);
        }
    };

    template <typename V, typename T, typename Abi>
    struct vector_pack_store<V, Vc::Vector<T, Abi>>
    {
        template <typename Iter>
        static void aligned(V const& value, Iter const& iter)
        {
            *iter = value;
        }

        template <typename Iter>
        static void unaligned(V const& value, Iter const& iter)
        {
            *iter = value;
        }
    };

    template <typename Value, typename T, std::size_t N, typename V,
        std::size_t W>
    struct vector_pack_store<Value, Vc::SimdArray<T, N, V, W>>
    {
        template <typename Iter>
        static void aligned(Value const& value, Iter const& iter)
        {
            *iter = value;
        }

        template <typename Iter>
        static void unaligned(Value const& value, Iter const& iter)
        {
            *iter = value;
        }
    };
}    // namespace hpx::parallel::traits

#endif
