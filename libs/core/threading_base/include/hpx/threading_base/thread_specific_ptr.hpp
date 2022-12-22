//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This code has been partially adopted from the Boost.Threads library
//
// (C) Copyright 2008 Anthony Williams
// (C) Copyright 2011-2012 Vicente J. Botet Escriba

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
#include <hpx/coroutines/detail/tss.hpp>
#include <hpx/threading_base/thread_data.hpp>

#include <memory>

namespace hpx::threads {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class thread_specific_ptr
    {
    private:
        typedef coroutines::detail::tss_cleanup_function cleanup_function;

        thread_specific_ptr(thread_specific_ptr const&) = delete;
        thread_specific_ptr& operator=(thread_specific_ptr const&) = delete;

        struct delete_data : coroutines::detail::tss_cleanup_function
        {
            void operator()(void* data) override
            {
                delete static_cast<T*>(data);
            }
        };

        struct run_custom_cleanup_function
          : coroutines::detail::tss_cleanup_function
        {
            explicit run_custom_cleanup_function(void (*cleanup_function_)(T*))
              : cleanup_function(cleanup_function_)
            {
            }

            void operator()(void* data) override
            {
                cleanup_function(static_cast<T*>(data));
            }

            void (*cleanup_function)(T*);
        };

        std::shared_ptr<cleanup_function> cleanup_;

    public:
        using element_type = T;

        thread_specific_ptr()
          : cleanup_(std::make_shared<delete_data>())
        {
        }

        explicit thread_specific_ptr(void (*func_)(T*))
        {
            if (func_)
                cleanup_.reset(new run_custom_cleanup_function(func_));
        }

        ~thread_specific_ptr()
        {
            // clean up data if this type is used locally for one thread
            if (get_self_ptr())
                coroutines::detail::erase_tss_node(this, true);
        }

        T* get() const
        {
            return static_cast<T*>(coroutines::detail::get_tss_data(this));
        }

        T* operator->() const
        {
            return get();
        }

        T& operator*() const
        {
            return *get();
        }

        T* release()
        {
            T* const temp = get();
            coroutines::detail::set_tss_data(
                this, std::shared_ptr<cleanup_function>());
            return temp;
        }

        void reset(T* new_value = nullptr)
        {
            T* const current_value = get();
            if (current_value != new_value)
            {
                coroutines::detail::set_tss_data(
                    this, cleanup_, new_value, true);
            }
        }
    };
}    // namespace hpx::threads

#endif
