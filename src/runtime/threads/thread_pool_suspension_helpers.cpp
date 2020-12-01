//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/async_local/apply.hpp>
#include <hpx/async_local/async.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/runtime/threads/thread_pool_suspension_helpers.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace threads {
    hpx::future<void> resume_processing_unit(
        thread_pool_base& pool, std::size_t virt_core)
    {
        if (!threads::get_self_ptr())
        {
            HPX_THROW_EXCEPTION(invalid_status, "resume_processing_unit",
                "cannot call resume_processing_unit from outside HPX, use"
                "resume_processing_unit_cb instead");
        }
        else if (!pool.get_scheduler()->has_scheduler_mode(
                     policies::enable_elasticity))
        {
            return hpx::make_exceptional_future<void>(
                HPX_GET_EXCEPTION(invalid_status, "resume_processing_unit",
                    "this thread pool does not support suspending "
                    "processing units"));
        }

        return hpx::async([&pool, virt_core]() -> void {
            return pool.resume_processing_unit_direct(virt_core, throws);
        });
    }

    void resume_processing_unit_cb(thread_pool_base& pool,
        util::function_nonser<void(void)> callback, std::size_t virt_core,
        error_code& ec)
    {
        if (!pool.get_scheduler()->has_scheduler_mode(
                policies::enable_elasticity))
        {
            HPX_THROWS_IF(ec, invalid_status, "resume_processing_unit_cb",
                "this thread pool does not support suspending "
                "processing units");
            return;
        }

        auto resume_direct_wrapper = [&pool, virt_core,
                                         callback = std::move(callback)]() {
            pool.resume_processing_unit_direct(virt_core, throws);
            callback();
        };

        if (threads::get_self_ptr())
        {
            hpx::apply(std::move(resume_direct_wrapper));
        }
        else
        {
            std::thread(std::move(resume_direct_wrapper)).detach();
        }
    }

    hpx::future<void> suspend_processing_unit(
        thread_pool_base& pool, std::size_t virt_core)
    {
        if (!threads::get_self_ptr())
        {
            HPX_THROW_EXCEPTION(invalid_status, "suspend_processing_unit",
                "cannot call suspend_processing_unit from outside HPX, use"
                "suspend_processing_unit_cb instead");
        }
        else if (!pool.get_scheduler()->has_scheduler_mode(
                     policies::enable_elasticity))
        {
            return hpx::make_exceptional_future<void>(
                HPX_GET_EXCEPTION(invalid_status, "suspend_processing_unit",
                    "this thread pool does not support suspending "
                    "processing units"));
        }
        else if (!pool.get_scheduler()->has_scheduler_mode(
                     policies::enable_stealing) &&
            hpx::this_thread::get_pool() == &pool)
        {
            return hpx::make_exceptional_future<void>(
                HPX_GET_EXCEPTION(invalid_status, "suspend_processing_unit",
                    "this thread pool does not support suspending "
                    "processing units from itself (no thread stealing)"));
        }

        return hpx::async([&pool, virt_core]() -> void {
            return pool.suspend_processing_unit_direct(virt_core, throws);
        });
    }

    void suspend_processing_unit_cb(thread_pool_base& pool,
        util::function_nonser<void(void)> callback, std::size_t virt_core,
        error_code& ec)
    {
        if (!pool.get_scheduler()->has_scheduler_mode(
                policies::enable_elasticity))
        {
            HPX_THROWS_IF(ec, invalid_status, "suspend_processing_unit_cb",
                "this thread pool does not support suspending processing "
                "units");
            return;
        }

        auto suspend_direct_wrapper = [&pool, virt_core,
                                          callback = std::move(callback)]() {
            pool.suspend_processing_unit_direct(virt_core, throws);
            callback();
        };

        if (threads::get_self_ptr())
        {
            if (!pool.get_scheduler()->has_scheduler_mode(
                    policies::enable_stealing) &&
                hpx::this_thread::get_pool() == &pool)
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "suspend_processing_unit_"
                    "cb",
                    "this thread pool does not support suspending "
                    "processing units from itself (no thread stealing)");
            }

            hpx::apply(std::move(suspend_direct_wrapper));
        }
        else
        {
            std::thread(std::move(suspend_direct_wrapper)).detach();
        }
    }

    future<void> resume_pool(thread_pool_base& pool)
    {
        if (!threads::get_self_ptr())
        {
            HPX_THROW_EXCEPTION(invalid_status, "resume_pool",
                "cannot call resume_pool from outside HPX, use resume_pool_cb "
                "or the member function resume_direct instead");
            return hpx::make_ready_future();
        }

        return hpx::async(
            [&pool]() -> void { return pool.resume_direct(throws); });
    }

    void resume_pool_cb(thread_pool_base& pool,
        util::function_nonser<void(void)> callback, error_code& /* ec */)
    {
        auto resume_direct_wrapper =
            [&pool, callback = std::move(callback)]() -> void {
            pool.resume_direct(throws);
            callback();
        };

        if (threads::get_self_ptr())
        {
            hpx::apply(std::move(resume_direct_wrapper));
        }
        else
        {
            std::thread(std::move(resume_direct_wrapper)).detach();
        }
    }

    future<void> suspend_pool(thread_pool_base& pool)
    {
        if (!threads::get_self_ptr())
        {
            HPX_THROW_EXCEPTION(invalid_status, "suspend_pool",
                "cannot call suspend_pool from outside HPX, use "
                "suspend_pool_cb or the member function suspend_direct "
                "instead");
            return hpx::make_ready_future();
        }
        else if (threads::get_self_ptr() &&
            hpx::this_thread::get_pool() == &pool)
        {
            return hpx::make_exceptional_future<void>(
                HPX_GET_EXCEPTION(bad_parameter, "suspend_pool",
                    "cannot suspend a pool from itself"));
        }

        return hpx::async(
            [&pool]() -> void { return pool.suspend_direct(throws); });
    }

    void suspend_pool_cb(thread_pool_base& pool,
        util::function_nonser<void(void)> callback, error_code& ec)
    {
        if (threads::get_self_ptr() && hpx::this_thread::get_pool() == &pool)
        {
            HPX_THROWS_IF(ec, bad_parameter, "suspend_pool_cb",
                "cannot suspend a pool from itself");
            return;
        }

        auto suspend_direct_wrapper = [&pool,
                                          callback = std::move(callback)]() {
            pool.suspend_direct(throws);
            callback();
        };

        if (threads::get_self_ptr())
        {
            hpx::apply(std::move(suspend_direct_wrapper));
        }
        else
        {
            std::thread(std::move(suspend_direct_wrapper)).detach();
        }
    }

}}    // namespace hpx::threads
