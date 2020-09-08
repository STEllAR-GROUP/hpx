//  Copyright (c) 2006 Joao Abecasis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

#if !defined(HPX_GCC_VERSION) && !defined(HPX_CLANG_VERSION) &&                \
    !(HPX_INTEL_VERSION > 1200 && !defined(HPX_WINDOWS))
#include <memory>    // for placement new
#include <mutex>
#endif

// clang-format off
#if !defined(HPX_WINDOWS)
#  define HPX_EXPORT_STATIC_ HPX_CORE_EXPORT
#else
#  define HPX_EXPORT_STATIC_
#endif
// clang-format on

namespace hpx { namespace util {
#if defined(HPX_GCC_VERSION) || defined(HPX_CLANG_VERSION) ||                  \
    (HPX_INTEL_VERSION > 1200 && !defined(HPX_WINDOWS)) || defined(HPX_MSVC)

    //
    // C++11 requires thread-safe initialization of function-scope statics.
    // For conforming compilers, we utilize this feature.
    //
    template <typename T, typename Tag = T>
    struct HPX_EXPORT_STATIC_ static_
    {
    public:
        HPX_NON_COPYABLE(static_);

    public:
        typedef T value_type;

        typedef T& reference;
        typedef T const& const_reference;

        static_()
        {
            get_reference();
        }

        operator reference()
        {
            return get();
        }

        operator const_reference() const
        {
            return get();
        }

        reference get()
        {
            return get_reference();
        }

        const_reference get() const
        {
            return get_reference();
        }

    private:
        static reference get_reference()
        {
            static T t;
            return t;
        }
    };

#else

    //
    //  Provides thread-safe initialization of a single static instance of T.
    //
    //  This instance is guaranteed to be constructed on static storage in a
    //  thread-safe manner, on the first call to the constructor of static_.
    //
    //  Requirements:
    //      T is default constructible or has one argument
    //      T::T() MUST not throw!
    //          this is a requirement of boost::call_once.
    //
    template <typename T, typename Tag = T>
    struct HPX_EXPORT_STATIC_ static_
    {
    public:
        HPX_NON_COPYABLE(static_);

    public:
        typedef T value_type;

    private:
        struct destructor
        {
            ~destructor()
            {
                static_::get_address()->~value_type();
            }
        };

        struct default_constructor
        {
            static void construct()
            {
                new (static_::get_address()) value_type();
                static destructor d;
            }
        };

    public:
        typedef T& reference;
        typedef T const& const_reference;

        static_()
        {
            std::call_once(constructed_, &default_constructor::construct);
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
            return reinterpret_cast<pointer>(data_);
        }

        typedef typename std::aligned_storage<sizeof(value_type),
            std::alignment_of<value_type>::value>::type storage_type;

        static storage_type data_;
        static std::once_flag constructed_;
    };

    template <typename T, typename Tag>
    typename static_<T, Tag>::storage_type static_<T, Tag>::data_;

    template <typename T, typename Tag>
    std::once_flag static_<T, Tag>::constructed_;
#endif
}}    // namespace hpx::util
