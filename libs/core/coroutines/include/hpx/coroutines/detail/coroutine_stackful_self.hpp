//  Copyright (c) 2019-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/detail/coroutine_impl.hpp>
#include <hpx/coroutines/detail/coroutine_self.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/functional/experimental/scope_exit.hpp>

#include <cstddef>
#include <limits>
#include <utility>

namespace hpx::threads::coroutines::detail {

    class coroutine_stackful_self : public coroutine_self
    {
    public:
        explicit coroutine_stackful_self(
            coroutine_impl* pimpl, coroutine_self* next_self = nullptr) noexcept
          : coroutine_self(next_self)
          , pimpl_(pimpl)
        {
            HPX_ASSERT(pimpl_);
        }

        arg_type yield_impl(result_type arg) override
        {
            this->pimpl_->bind_result(arg);

            {
                // store the current this and write it back to the TSS on exit
                set_self(this->next_self_);
                auto on_exit =
                    hpx::experimental::scope_exit([this] { set_self(this); });

                this->pimpl_->yield();
            }

            return *pimpl_->args();
        }

        thread_id_type get_thread_id() const noexcept override
        {
            return pimpl_->get_thread_id();
        }

#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
        std::size_t get_thread_phase() const noexcept override
        {
            return pimpl_->get_thread_phase();
        }
#else
        std::size_t get_thread_phase() const noexcept override
        {
            return 0;
        }
#endif

#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
        std::ptrdiff_t get_available_stack_space() const noexcept override
        {
            return pimpl_->get_available_stack_space();
        }
#else
        std::ptrdiff_t get_available_stack_space() const noexcept override
        {
            return (std::numeric_limits<std::ptrdiff_t>::max)();
        }
#endif

        std::size_t get_thread_data() const noexcept override
        {
            return pimpl_->get_thread_data();
        }
        std::size_t set_thread_data(std::size_t data) noexcept override
        {
            return pimpl_->set_thread_data(data);
        }

#if defined(HPX_HAVE_LIBCDS)
        std::size_t get_libcds_data() const override
        {
            return pimpl_->get_libcds_data();
        }
        std::size_t set_libcds_data(std::size_t data) override
        {
            return pimpl_->set_libcds_data(data);
        }

        std::size_t get_libcds_hazard_pointer_data() const override
        {
            return pimpl_->get_libcds_hazard_pointer_data();
        }
        std::size_t set_libcds_hazard_pointer_data(std::size_t data) override
        {
            return pimpl_->set_libcds_hazard_pointer_data(data);
        }

        std::size_t get_libcds_dynamic_hazard_pointer_data() const override
        {
            return pimpl_->get_libcds_dynamic_hazard_pointer_data();
        }
        std::size_t set_libcds_dynamic_hazard_pointer_data(
            std::size_t data) override
        {
            return pimpl_->set_libcds_dynamic_hazard_pointer_data(data);
        }
#endif

#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
        tss_storage* get_thread_tss_data() override
        {
            return pimpl_->get_thread_tss_data(false);
        }
#else
        tss_storage* get_thread_tss_data() override
        {
            return nullptr;
        }
#endif

#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
        tss_storage* get_or_create_thread_tss_data() override
        {
            return pimpl_->get_thread_tss_data(true);
        }
#else
        tss_storage* get_or_create_thread_tss_data() override
        {
            return nullptr;
        }
#endif

        std::size_t& get_continuation_recursion_count() noexcept override
        {
            return pimpl_->get_continuation_recursion_count();
        }

    protected:
        coroutine_impl* get_impl() noexcept override
        {
            return pimpl_;
        }
        coroutine_impl* pimpl_;
    };
}    // namespace hpx::threads::coroutines::detail
