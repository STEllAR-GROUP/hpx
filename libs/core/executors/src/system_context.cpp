//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/executors/execution_parameters_fwd.hpp>
#include <hpx/executors/system_context.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>

namespace hpx::execution::experimental {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        void intrusive_ptr_add_ref(system_scheduler_base* p) noexcept
        {
            ++p->count;
        }

        void intrusive_ptr_release(system_scheduler_base* p) noexcept
        {
            if (0 == --p->count)
                delete p;
        }

        ///////////////////////////////////////////////////////////////////////
        struct system_scheduler_impl : detail::system_scheduler_base
        {
            ~system_scheduler_impl() = default;

            void execute(hpx::move_only_function<void()> set_value,
                hpx::move_only_function<void(std::exception_ptr)>
                    set_error) & noexcept override
            {
                auto f = [set_value = HPX_MOVE(set_value),
                             set_error = HPX_MOVE(set_error)]() {
                    hpx::detail::try_catch_exception_ptr(set_value, set_error);
                };

                scheduler.execute(HPX_MOVE(f));
            }

            std::size_t max_concurrency() const noexcept override
            {
                return hpx::parallel::execution::processing_units_count(
                    scheduler);
            }

            thread_pool_policy_scheduler<hpx::launch> scheduler;
        };
    }    // namespace detail

    system_context::system_context()
      : ctx(new detail::system_scheduler_impl())
    {
    }

    system_context::~system_context() = default;
}    // namespace hpx::execution::experimental
