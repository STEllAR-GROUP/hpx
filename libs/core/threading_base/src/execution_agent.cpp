//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2020-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/lock_registration/detail/register_locks.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/threading_base/execution_agent.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/set_thread_state.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>

#ifdef HPX_HAVE_THREAD_DESCRIPTION
#include <hpx/threading_base/detail/reset_lco_description.hpp>
#endif
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
#include <hpx/debugging/backtrace.hpp>
#include <hpx/threading_base/detail/reset_backtrace.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace hpx::threads {

    execution_agent::execution_agent(
        coroutines::detail::coroutine_impl* coroutine) noexcept
      : self_(coroutine)
    {
    }

    std::string execution_agent::description() const
    {
        thread_id_type const id = self_.get_thread_id();
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROW_EXCEPTION(hpx::error::null_thread_id,
                "execution_agent::description",
                "null thread id encountered (is this executed on a "
                "HPX-thread?)");
        }

        return hpx::util::format(
            "{}: {}", id, get_thread_id_data(id)->get_description());
    }

    void execution_agent::yield(char const* desc)
    {
        do_yield(desc, hpx::threads::thread_schedule_state::pending);
    }

    void execution_agent::yield_k(std::size_t k, char const* desc)
    {
        if (k < 4)    //-V112
        {
        }
#if defined(HPX_SMT_PAUSE)
        else if (k < 16)
        {
            HPX_SMT_PAUSE;
        }
#endif
        else if (k < 32 || k & 1)    //-V112
        {
            do_yield(desc, hpx::threads::thread_schedule_state::pending_boost);
        }
        else
        {
            do_yield(desc, hpx::threads::thread_schedule_state::pending);
        }
    }

    void execution_agent::resume(
        hpx::threads::thread_priority priority, char const* desc)
    {
        do_resume(priority, desc, threads::thread_restart_state::signaled);
    }

    void execution_agent::abort(char const* desc)
    {
        do_resume(hpx::threads::thread_priority::default_, desc,
            threads::thread_restart_state::abort);
    }

    void execution_agent::suspend(char const* desc)
    {
        do_yield(desc, threads::thread_schedule_state::suspended);
    }

    void execution_agent::sleep_for(
        hpx::chrono::steady_duration const& sleep_duration, char const* desc)
    {
        sleep_until(sleep_duration.from_now(), desc);
    }

    void execution_agent::sleep_until(
        hpx::chrono::steady_time_point const& sleep_time, char const* desc)
    {
        // Just yield until time has passed by...
        auto now = std::chrono::steady_clock::now();

        // Note: we yield at least once to allow for other threads to make
        // progress in any case. We also use yield instead of yield_k for the
        // same reason.
        std::size_t k = 0;
        do
        {
            if (k < 32 || k & 1)    //-V112
            {
                do_yield(
                    desc, hpx::threads::thread_schedule_state::pending_boost);
            }
            else
            {
                do_yield(desc, hpx::threads::thread_schedule_state::pending);
            }
            ++k;
            now = std::chrono::steady_clock::now();
        } while (now < sleep_time.value());
    }

    hpx::threads::thread_restart_state execution_agent::do_yield(
        char const* desc, threads::thread_schedule_state state)
    {
        thread_id_ref_type id = self_.get_thread_id();    // keep alive
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROW_EXCEPTION(hpx::error::null_thread_id,
                "execution_agent::do_yield",
                "null thread id encountered (is this executed on a "
                "HPX-thread?)");
        }

        // handle interruption, if needed
        thread_data* thrd_data = get_thread_id_data(id);
        if (HPX_UNLIKELY(thrd_data == nullptr))
        {
            HPX_THROW_EXCEPTION(hpx::error::null_thread_id,
                "execution_agent::do_yield",
                "null thread id encountered (is this executed on a "
                "HPX-thread?)");
        }

        thrd_data->interruption_point();

        thrd_data->set_last_worker_thread_num(
            hpx::get_local_worker_thread_num());

        threads::thread_restart_state statex;

        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            [[maybe_unused]] threads::detail::reset_lco_description reset_desc(
                id.noref(), threads::thread_description(desc));
#endif
#if defined(HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION)
            [[maybe_unused]] threads::detail::reset_backtrace reset_bt(
                id.noref());
#endif
#if defined(HPX_HAVE_VERIFY_LOCKS)
            [[maybe_unused]] auto held_locks = hpx::experimental::scope_exit(
                [data = hpx::util::get_held_locks_data()]() mutable {
                    hpx::util::set_held_locks_data(HPX_MOVE(data));
                });
#endif
            HPX_ASSERT(thrd_data != nullptr &&
                thrd_data->get_state().state() ==
                    thread_schedule_state::active);
            HPX_ASSERT(state != thread_schedule_state::active);

            // actual yield operation
            statex = self_.yield(
                threads::thread_result_type(state, threads::invalid_thread_id));

            HPX_ASSERT(thrd_data != nullptr &&
                thrd_data->get_state().state() ==
                    thread_schedule_state::active);
        }

        // handle interruption, if needed
        thrd_data->interruption_point();

        // handle interrupt and abort
        if (statex == threads::thread_restart_state::abort)
        {
            HPX_THROW_EXCEPTION(hpx::error::yield_aborted, desc,
                "thread({}) aborted (yield returned wait_abort)",
                description());
        }

        return statex;
    }

    void execution_agent::do_resume(hpx::threads::thread_priority priority,
        char const* /* desc */, hpx::threads::thread_restart_state statex) const
    {
        threads::detail::set_thread_state(self_.get_thread_id(),
            thread_schedule_state::pending, statex, priority,
            thread_schedule_hint{}, false);
    }
}    // namespace hpx::threads
