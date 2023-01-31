////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>

// Needed to potentially get _GLIBCXX_HAVE_TLS
#include <cstdlib>

// clang-format off
#if !defined(HPX_WINDOWS)
#  define HPX_CORE_EXPORT_THREAD_SPECIFIC_PTR HPX_CORE_EXPORT
#else
#  define HPX_CORE_EXPORT_THREAD_SPECIFIC_PTR
#endif
// clang-format on

#if !defined(__ANDROID__) && !defined(ANDROID) && !defined(__bgq__)

namespace hpx::util {

    template <typename T, typename Tag>
    struct HPX_CORE_EXPORT_THREAD_SPECIFIC_PTR thread_specific_ptr
    {
        using element_type = T;

        [[nodiscard]] T* get() const noexcept
        {
            return ptr_;
        }

        [[nodiscard]] T* operator->() const noexcept
        {
            return ptr_;
        }

        [[nodiscard]] T& operator*() const noexcept
        {
            HPX_ASSERT(nullptr != ptr_);
            return *ptr_;
        }

        void reset(T* new_value = nullptr) noexcept
        {
            delete ptr_;
            ptr_ = new_value;
        }

    private:
        static thread_local T* ptr_;
    };

    template <typename T, typename Tag>
    thread_local T* thread_specific_ptr<T, Tag>::ptr_ = nullptr;
}    // namespace hpx::util

#else

#include <hpx/type_support/static.hpp>
#include <pthread.h>

namespace hpx::util {

    namespace detail {

        struct thread_specific_ptr_key
        {
            thread_specific_ptr_key()
            {
                // pthread_once(&key_once, &thread_specific_ptr_key::make_key);
                pthread_key_create(&key, nullptr);
            }

            pthread_key_t key;
        };
    }    // namespace detail

    template <typename T, typename Tag>
    struct HPX_CORE_EXPORT_THREAD_SPECIFIC_PTR thread_specific_ptr
    {
        using element_type = T;

        [[nodiscard]] static pthread_key_t get_key()
        {
            static_<detail::thread_specific_ptr_key,
                thread_specific_ptr<T, Tag>>
                key_holder;

            return key_holder.get().key;
        }

        [[nodiscard]] T* get() const
        {
            return reinterpret_cast<T*>(
                pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));
        }

        [[nodiscard]] T* operator->() const
        {
            return reinterpret_cast<T*>(
                pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));
        }

        [[nodiscard]] T& operator*() const
        {
            T* ptr = nullptr;

            ptr = reinterpret_cast<T*>(
                pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));
            HPX_ASSERT(nullptr != ptr);
            return *ptr;
        }

        void reset(T* new_value = nullptr)
        {
            T* ptr = nullptr;

            ptr = reinterpret_cast<T*>(
                pthread_getspecific(thread_specific_ptr<T, Tag>::get_key()));
            if (nullptr != ptr)
                delete ptr;

            ptr = new_value;
            pthread_setspecific(thread_specific_ptr<T, Tag>::get_key(), ptr);
        }
    };
}    // namespace hpx::util

#endif
