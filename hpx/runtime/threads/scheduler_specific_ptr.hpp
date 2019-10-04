//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This code has been partially adopted from the Boost.Threads library
//
// (C) Copyright 2008 Anthony Williams
// (C) Copyright 2011-2012 Vicente J. Botet Escriba

#if !defined(HPX_RUNTIME_THREADS_SCHEDULER_TSS_AUG_08_2015_0733PM)
#define HPX_RUNTIME_THREADS_SCHEDULER_TSS_AUG_08_2015_0733PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/coroutines/detail/tss.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>

#include <memory>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        HPX_API_EXPORT void set_tss_data(void const* key,
            std::shared_ptr<
                coroutines::detail::tss_cleanup_function
            > const& func,
            void* tss_data, bool cleanup_existing);
        HPX_API_EXPORT void* get_tss_data(void const* key);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class scheduler_specific_ptr
    {
    public:
        HPX_NON_COPYABLE(scheduler_specific_ptr);

    private:
        struct delete_data : coroutines::detail::tss_cleanup_function
        {
            void operator()(void* data)
            {
                delete static_cast<T*>(data);
            }
        };

        struct run_custom_cleanup_function
          : coroutines::detail::tss_cleanup_function
        {
            void (*cleanup_function)(T*);

            explicit run_custom_cleanup_function(void (*cleanup_function_)(T*))
              : cleanup_function(cleanup_function_)
            {}

            void operator()(void* data)
            {
                cleanup_function(static_cast<T*>(data));
            }
        };

        std::shared_ptr<coroutines::detail::tss_cleanup_function> cleanup;

    public:
        typedef T element_type;

        scheduler_specific_ptr()
          : cleanup(std::make_shared<delete_data>())
        {}

        explicit scheduler_specific_ptr(void (*func_)(T*))
        {
            if (func_)
                cleanup.reset(new run_custom_cleanup_function(func_));
        }

        ~scheduler_specific_ptr()
        {
            // clean up data if this type is used locally for one thread
            if (get_self_ptr())
            {
                detail::set_tss_data(this,
                    std::shared_ptr<coroutines::detail::tss_cleanup_function>(),
                    0, true);
            }
        }

        T* get() const
        {
            return static_cast<T*>(detail::get_tss_data(this));
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
            detail::set_tss_data(this,
                std::shared_ptr<coroutines::detail::tss_cleanup_function>(),
                0, false);
            return temp;
        }

        void reset(T* new_value = nullptr)
        {
            T* const current_value = get();
            if (current_value != new_value)
                detail::set_tss_data(this, cleanup, new_value, true);
        }
    };
}}

#endif
