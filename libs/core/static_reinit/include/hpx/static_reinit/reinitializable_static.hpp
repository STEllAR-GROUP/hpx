//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2006 Joao Abecasis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/static_reinit/static_reinit.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <memory>    // for placement new
#include <mutex>
#include <type_traits>

// clang-format off
#if !defined(HPX_WINDOWS)
#  define HPX_EXPORT_REINITIALIZABLE_STATIC HPX_CORE_EXPORT
#else
#  define HPX_EXPORT_REINITIALIZABLE_STATIC
#endif
// clang-format on

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    //  Provides thread-safe initialization of a single static instance of T.
    //
    //  This instance is guaranteed to be constructed on static storage in a
    //  thread-safe manner, on the first call to the constructor of static_.
    //
    //  Requirements:
    //      T is default constructible or has one argument
    //
    //  In addition this type registers global construction and destruction
    //  functions used by the HPX runtime system to reinitialize the held data
    //  structures.
    template <typename T, typename Tag = T, std::size_t N = 1>
    struct HPX_EXPORT_REINITIALIZABLE_STATIC reinitializable_static;

    //////////////////////////////////////////////////////////////////////////
    template <typename T, typename Tag, std::size_t N>
    struct HPX_EXPORT_REINITIALIZABLE_STATIC reinitializable_static
    {
    public:
        HPX_NON_COPYABLE(reinitializable_static);

    public:
        typedef T value_type;

    private:
        static void default_construct()
        {
            for (std::size_t i = 0; i < N; ++i)
                new (get_address(i)) value_type();
        }

        template <typename U>
        static void value_construct(U const& v)
        {
            for (std::size_t i = 0; i < N; ++i)
                new (get_address(i)) value_type(v);
        }

        static void destruct()
        {
            for (std::size_t i = 0; i < N; ++i)
                get_address(i)->~value_type();
        }

        ///////////////////////////////////////////////////////////////////////
        static void default_constructor()
        {
            default_construct();
            reinit_register(&reinitializable_static::default_construct,
                &reinitializable_static::destruct);
        }

        template <typename U>
        static void value_constructor(U const* pv)
        {
            value_construct(*pv);
            reinit_register(
                util::bind_front(
                    &reinitializable_static::template value_construct<U>, *pv),
                &reinitializable_static::destruct);
        }

    public:
        typedef T& reference;
        typedef T const& const_reference;

        reinitializable_static()
        {
#if !defined(__CUDACC__)
            // do not rely on ADL to find the proper call_once
            std::call_once(
                constructed_, &reinitializable_static::default_constructor);
#endif
        }

        template <typename U>
        reinitializable_static(U const& val)
        {
#if !defined(__CUDACC__)
            // do not rely on ADL to find the proper call_once
            std::call_once(constructed_,
                util::bind_front(
                    &reinitializable_static::template value_constructor<U>,
                    const_cast<U const*>(std::addressof(val))));
#else
            HPX_UNUSED(val);
#endif
        }

        operator reference()
        {
            return this->get();
        }

        operator const_reference() const
        {
            return this->get();
        }

        reference get(std::size_t item = 0)
        {
            return *this->get_address(item);
        }

        const_reference get(std::size_t item = 0) const
        {
            return *this->get_address(item);
        }

    private:
        typedef typename std::add_pointer<value_type>::type pointer;

        static pointer get_address(std::size_t item)
        {
            HPX_ASSERT(item < N);
            return reinterpret_cast<pointer>(data_ + item);
        }

        typedef typename std::aligned_storage<sizeof(value_type),
            std::alignment_of<value_type>::value>::type storage_type;

        static storage_type data_[N];
        static std::once_flag constructed_;
    };

    template <typename T, typename Tag, std::size_t N>
    typename reinitializable_static<T, Tag, N>::storage_type
        reinitializable_static<T, Tag, N>::data_[N];

    template <typename T, typename Tag, std::size_t N>
    std::once_flag reinitializable_static<T, Tag, N>::constructed_;
}}    // namespace hpx::util

#undef HPX_EXPORT_REINITIALIZABLE_STATIC
