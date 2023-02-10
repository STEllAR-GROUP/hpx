//  Copyright (c) 2006 Joao Abecasis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <memory>
#include <type_traits>

#if !defined(HPX_GCC_VERSION) && !defined(HPX_CLANG_VERSION) &&                \
    !(HPX_INTEL_VERSION > 1200 && !defined(HPX_WINDOWS))
#include <mutex>
#endif

// clang-format off
#if !defined(HPX_WINDOWS)
#  define HPX_EXPORT_STATIC_ HPX_CORE_EXPORT
#else
#  define HPX_EXPORT_STATIC_
#endif
// clang-format on

namespace hpx::util {

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

        ~static_() = default;

    public:
        using value_type = T;
        using reference = T&;
        using const_reference = T const&;

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

        [[nodiscard]] reference get()
        {
            return get_reference();
        }

        [[nodiscard]] const_reference get() const
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

        ~static_() = default;

    public:
        using value_type = T;

    private:
        struct destructor
        {
            ~destructor()
            {
                std::destroy_at(static_::get_address());
            }
        };

        struct default_constructor
        {
            static void construct()
            {
                hpx::construct_at(static_::get_address());
                static destructor d;
            }
        };

    public:
        using reference = T&;
        using const_reference = T const&;

        static_()
        {
            std::call_once(constructed_, &default_constructor::construct);
        }

        operator reference() noexcept
        {
            return this->get();
        }

        operator const_reference() const noexcept
        {
            return this->get();
        }

        [[nodiscard]] reference get() noexcept
        {
            return *this->get_address();
        }

        [[nodiscard]] const_reference get() const noexcept
        {
            return *this->get_address();
        }

    private:
        using pointer = std::add_pointer_t<value_type>;

        [[nodiscard]] static pointer get_address() noexcept
        {
            return reinterpret_cast<pointer>(data_);
        }

        using storage_type = std::aligned_storage_t<sizeof(value_type),
            std::alignment_of_v<value_type>>;

        static storage_type data_;
        static std::once_flag constructed_;
    };

    template <typename T, typename Tag>
    typename static_<T, Tag>::storage_type static_<T, Tag>::data_;

    template <typename T, typename Tag>
    std::once_flag static_<T, Tag>::constructed_;
#endif
}    // namespace hpx::util
