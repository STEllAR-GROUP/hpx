//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/executors/execution_parameters_fwd.hpp>
#include <hpx/executors/detail/bulk_system_context.hpp>
#include <hpx/executors/system_context.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>

namespace hpx::execution::experimental { namespace detail {

    ///////////////////////////////////////////////////////////////////////
    void intrusive_ptr_add_ref(bulk_system_scheduler_base* p) noexcept
    {
        ++p->count;
    }

    void intrusive_ptr_release(bulk_system_scheduler_base* p) noexcept
    {
        if (0 == --p->count)
            delete p;
    }

    ///////////////////////////////////////////////////////////////////////
    struct bulk_system_scheduler_impl : bulk_system_scheduler_base
    {
        bulk_system_scheduler_impl(system_scheduler const&) {}

        ~bulk_system_scheduler_impl() = default;

        void bulk_set_value(hpx::move_only_function<void()> set_value) override
        {
        }
        void bulk_set_error(hpx::move_only_function<void()> set_error) override
        {
        }
        void bulk_set_stopped(
            hpx::move_only_function<void()> set_stopped) override
        {
        }

        thread_pool_policy_scheduler<hpx::launch> scheduler;
    };

    hpx::intrusive_ptr<bulk_system_scheduler_base> get_bulk_context(
        system_scheduler const& s)
    {
        return {new bulk_system_scheduler_impl(s)};
    }
}}    // namespace hpx::execution::experimental::detail
