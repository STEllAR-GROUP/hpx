//  Copyright (c) 2019-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/coroutines/detail/coroutine_impl.hpp>
#include <hpx/coroutines/detail/coroutine_stackful_self.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/coroutines/thread_id_type.hpp>

#include <cstddef>
#include <limits>
#include <utility>

namespace hpx { namespace threads { namespace coroutines { namespace detail {

    class coroutine_stackful_self_direct : public coroutine_stackful_self
    {
    public:
        explicit coroutine_stackful_self_direct(
            coroutine_impl* pimpl, coroutine_self* next_self)
          : coroutine_stackful_self(pimpl, next_self)
          , next_self_(next_self)
        {
            HPX_ASSERT(next_self_);
        }

        // direct execution of a thread needs to use the executing context for
        // yielding
        arg_type yield_impl(result_type arg) override
        {
            return next_self_->yield_impl(arg);
        }

        thread_id_type get_outer_thread_id() const override
        {
            return next_self_->get_outer_thread_id();
        }

#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
        std::size_t get_thread_phase() const override
        {
            return next_self_->get_thread_phase();
        }
#endif

#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
        // return the executing thread's available stack space
        std::ptrdiff_t get_available_stack_space() override
        {
            return next_self_->get_available_stack_space();
        }
#endif

        // return the executing thread's recursion count
        std::size_t& get_continuation_recursion_count() override
        {
            return next_self_->get_continuation_recursion_count();
        }

    private:
        // if we chain direct calls the executing thread needs to be inherited
        // down
        coroutine_impl* get_impl() override
        {
            return coroutine_accessor::get_impl(*next_self_);
        }

        coroutine_self* next_self_;
    };
}}}}    // namespace hpx::threads::coroutines::detail
