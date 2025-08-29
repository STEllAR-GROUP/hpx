//  Copyright (c) 2007-2023 Hartmut Kaiser
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
#include <hpx/type_support/aligned_storage.hpp>
#include <hpx/type_support/construct_at.hpp>

#include <cstddef>
#include <memory>    // for placement new
#include <mutex>
#include <type_traits>

// clang-format off
#if !defined(HPX_WINDOWS)
#  define HPX_CORE_EXPORT_REINITIALIZABLE_STATIC HPX_CORE_EXPORT
#else
#  define HPX_CORE_EXPORT_REINITIALIZABLE_STATIC
#endif
// clang-format on

namespace hpx::util {

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
    struct HPX_CORE_EXPORT_REINITIALIZABLE_STATIC reinitializable_static;

    //////////////////////////////////////////////////////////////////////////
    template <typename T, typename Tag, std::size_t N>
    struct HPX_CORE_EXPORT_REINITIALIZABLE_STATIC reinitializable_static
    {
    public:
        HPX_NON_COPYABLE(reinitializable_static);

    public:
        using value_type = T;

    private:
        static void default_construct()
        {
            for (std::size_t i = 0; i < N; ++i)
                hpx::construct_at(get_address(i));
        }

        template <typename U>
        static void value_construct(U const& v)
        {
            for (std::size_t i = 0; i < N; ++i)
                hpx::construct_at(get_address(i), v);
        }

        static void destruct()
        {
            for (std::size_t i = 0; i < N; ++i)
                std::destroy_at(get_address(i));
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
                hpx::bind_front(
                    &reinitializable_static::value_construct<U>, *pv),
                &reinitializable_static::destruct);
        }

    public:
        using reference = T&;
        using const_reference = T const&;

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
                hpx::bind_front(&reinitializable_static::value_constructor<U>,
                    const_cast<U const*>(std::addressof(val))));
#else
            HPX_UNUSED(val);
#endif
        }

        operator reference() noexcept
        {
            return this->get();
        }

        operator const_reference() const noexcept
        {
            return this->get();
        }

        reference get(std::size_t item = 0) noexcept
        {
            return *this->get_address(item);
        }

        const_reference get(std::size_t item = 0) const noexcept
        {
            return *this->get_address(item);
        }

    private:
        using pointer = std::add_pointer_t<value_type>;

        static pointer get_address(std::size_t item) noexcept
        {
            HPX_ASSERT(item < N);
            return reinterpret_cast<pointer>(data_ + item);
        }

        using storage_type = hpx::aligned_storage_t<sizeof(value_type),
            std::alignment_of_v<value_type>>;

        static storage_type data_[N];
        static std::once_flag constructed_;
    };

    template <typename T, typename Tag, std::size_t N>
    typename reinitializable_static<T, Tag, N>::storage_type
        reinitializable_static<T, Tag, N>::data_[N];

    template <typename T, typename Tag, std::size_t N>
    std::once_flag reinitializable_static<T, Tag, N>::constructed_;
}    // namespace hpx::util

#undef HPX_CORE_EXPORT_REINITIALIZABLE_STATIC
