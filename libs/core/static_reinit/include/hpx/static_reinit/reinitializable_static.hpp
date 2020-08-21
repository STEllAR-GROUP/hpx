//  Copyright (c) 2007-2020 Hartmut Kaiser
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

#include <atomic>
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

    template <typename T, typename Tag = T>
    struct HPX_EXPORT_REINITIALIZABLE_STATIC reinitializable_static
    {
    public:
        HPX_NON_COPYABLE(reinitializable_static);

    public:
        typedef T value_type;

    private:
        static bool default_construct()
        {
            bool expected = false;
            if (needs_destruction().compare_exchange_strong(expected, true))
            {
                new (get_address()) value_type();
                return true;
            }
            return false;
        }

        template <typename U>
        static bool value_construct(U const& v)
        {
            bool expected = false;
            if (needs_destruction().compare_exchange_strong(expected, true))
            {
                new (get_address()) value_type(v);
                return true;
            }
            return false;
        }

        static void destruct()
        {
            bool expected = true;
            if (needs_destruction().compare_exchange_strong(expected, false))
            {
                get_address()->~value_type();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        static void default_constructor()
        {
            if (default_construct())
            {
                reinit_register(&reinitializable_static::default_construct,
                    &reinitializable_static::destruct);
            }
        }

        template <typename U>
        static void value_constructor(U const* pv)
        {
            if (value_construct(*pv))
            {
                reinit_register(
                    util::bind_front(
                        &reinitializable_static::template value_construct<U>,
                        *pv),
                    &reinitializable_static::destruct);
            }
        }

    public:
        typedef T& reference;
        typedef T const& const_reference;

        reinitializable_static()
        {
#if !defined(__CUDACC__)
            reinitializable_static::default_constructor();
#endif
        }

        template <typename U>
        reinitializable_static(U const& val)
        {
#if !defined(__CUDACC__)
            reinitializable_static::value_constructor(std::addressof(val));
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

        reference get()
        {
            return *this->get_address();
        }

        const_reference get() const
        {
            return *this->get_address();
        }

    private:
        typedef typename std::add_pointer<value_type>::type pointer;

        static pointer get_address()
        {
            return reinterpret_cast<pointer>(&data_);
        }

        typedef typename std::aligned_storage<sizeof(value_type),
            std::alignment_of<value_type>::value>::type storage_type;

        static storage_type data_;

        static std::atomic<bool>& needs_destruction()
        {
            static std::atomic<bool> needs_destruction(false);
            return needs_destruction;
        }
    };

    template <typename T, typename Tag>
    typename reinitializable_static<T, Tag>::storage_type
        reinitializable_static<T, Tag>::data_;

}}    // namespace hpx::util

#undef HPX_EXPORT_REINITIALIZABLE_STATIC
