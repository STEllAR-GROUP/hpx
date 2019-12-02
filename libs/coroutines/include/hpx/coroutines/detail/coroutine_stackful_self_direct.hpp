//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_COROUTINES_DETAIL_STACKFUL_SELF_DIRECT_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_DETAIL_STACKFUL_SELF_DIRECT_HPP

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/coroutines/detail/coroutine_impl.hpp>
#include <hpx/coroutines/detail/coroutine_self.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/functional/function.hpp>

#include <cstddef>
#include <exception>
#include <limits>
#include <utility>

namespace hpx { namespace threads { namespace coroutines { namespace detail {

    class coroutine_stackful_self_direct : public coroutine_self
    {
    public:
        explicit coroutine_stackful_self_direct(
            impl_type* pimpl, coroutine_self* next_self = nullptr)
          : coroutine_self(next_self)
          , pimpl_(pimpl)
          , pimpl_next_(coroutine_accessor::get_impl(*next_self))
        {
        }

        // direct execution of a thread needs to use the executing context for
        // yielding
        arg_type yield_impl(result_type arg) override
        {
            HPX_ASSERT(pimpl_next_);

            this->pimpl_next_->bind_result(arg);

            {
                reset_self_on_exit on_exit(this);
                this->pimpl_next_->yield();
            }

            return *pimpl_next_->args();
        }

        thread_id_type get_thread_id() const override
        {
            HPX_ASSERT(pimpl_);
            return pimpl_->get_thread_id();
        }

        std::size_t get_thread_phase() const override
        {
#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
            HPX_ASSERT(pimpl_next_);
            return pimpl_next_->get_thread_phase();
#else
            return 0;
#endif
        }

        // return the executing thread's available stack space
        std::ptrdiff_t get_available_stack_space() override
        {
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
            return pimpl_next_->get_available_stack_space();
#else
            return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
        }

        std::size_t get_thread_data() const override
        {
            HPX_ASSERT(pimpl_);
            return pimpl_->get_thread_data();
        }
        std::size_t set_thread_data(std::size_t data) override
        {
            HPX_ASSERT(pimpl_);
            return pimpl_->set_thread_data(data);
        }

        tss_storage* get_thread_tss_data() override
        {
#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
            HPX_ASSERT(pimpl_);
            return pimpl_->get_thread_tss_data(false);
#else
            return nullptr;
#endif
        }

        tss_storage* get_or_create_thread_tss_data() override
        {
#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
            HPX_ASSERT(pimpl_);
            return pimpl_->get_thread_tss_data(true);
#else
            return nullptr;
#endif
        }

        // return the executing thread's recursion count
        std::size_t& get_continuation_recursion_count() override
        {
            HPX_ASSERT(pimpl_next_);
            return pimpl_next_->get_continuation_recursion_count();
        }

    private:
        // if we chain direct calls the executing thread needs to be inherited
        // down
        coroutine_impl* get_impl() override
        {
            return pimpl_next_;
        }

        coroutine_impl* pimpl_;
        coroutine_impl* pimpl_next_;
    };
}}}}    // namespace hpx::threads::coroutines::detail

#endif /*HPX_RUNTIME_THREADS_COROUTINES_DETAIL_SELF_HPP*/
