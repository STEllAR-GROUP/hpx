//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/format.hpp>
#include <hpx/logging.hpp>
#include <hpx/coroutines/thread_enums.hpp>

#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/execution_agent.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/thread_description.hpp>

#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
#include <hpx/util/backtrace.hpp>
#endif

#include <cstddef>
#include <sstream>
#include <string>

namespace hpx { namespace threads {

    execution_agent::execution_agent(
        coroutines::detail::coroutine_impl* coroutine) noexcept
      : self_(coroutine)
    {
    }

    std::string execution_agent::description() const
    {
        thread_id_type id = self_.get_thread_id();
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROW_EXCEPTION(null_thread_id, "execution_agent::description",
                "null thread id encountered (is this executed on a "
                "HPX-thread?)");
        }

        return hpx::util::format(
            "{}: {}", id, get_thread_id_data(id)->get_description());
    }

    void execution_agent::yield(const char* desc)
    {
        do_yield(desc, hpx::threads::pending);
    }

    void execution_agent::yield_k(std::size_t k, const char* desc)
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
            do_yield(desc, hpx::threads::pending_boost);
        }
        else
        {
            do_yield(desc, hpx::threads::pending);
        }
    }

    void execution_agent::resume(const char* desc)
    {
        do_resume(desc, wait_signaled);
    }

    void execution_agent::abort(const char* desc)
    {
        do_resume(desc, wait_abort);
    }

    void execution_agent::suspend(const char* desc)
    {
        do_yield(desc, suspended);
    }

    void execution_agent::sleep_for(
        hpx::util::steady_duration const& sleep_duration, const char* desc)
    {
        sleep_until(sleep_duration.from_now(), desc);
    }

    void execution_agent::sleep_until(
        hpx::util::steady_time_point const& sleep_time, const char* desc)
    {
        auto now = hpx::util::steady_clock::now();

        // Just yield until time has passed by...
        for (std::size_t k = 0; now < sleep_time.value(); ++k)
        {
            yield_k(k, desc);
            now = hpx::util::steady_clock::now();
        }
    }

    namespace detail {
        struct reset_lco_description
        {
            reset_lco_description(threads::thread_id_type const& id,
                util::thread_description const& description)
              : id_(id)
            {
                old_desc_ =
                    threads::set_thread_lco_description(id_, description);
            }

            ~reset_lco_description()
            {
                threads::set_thread_lco_description(id_, old_desc_);
            }

            threads::thread_id_type id_;
            util::thread_description old_desc_;
        };
    }    // namespace detail

    hpx::threads::thread_state_ex_enum execution_agent::do_yield(
        const char* desc, threads::thread_state_enum state)
    {
        thread_id_type id = self_.get_thread_id();
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROW_EXCEPTION(null_thread_id, "execution_agent::do_yield",
                "null thread id encountered (is this executed on a "
                "HPX-thread?)");
        }

        // handle interruption, if needed
        threads::interruption_point(id);

        threads::thread_state_ex_enum statex = threads::wait_unknown;

        {
#ifdef HPX_HAVE_THREAD_DESCRIPTION
            detail::reset_lco_description desc(
                id, util::thread_description(desc));
#endif
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
            detail::reset_backtrace bt(id, ec);
#endif
            basic_execution::this_thread::suspend_agent on_exit;

            HPX_ASSERT(get_thread_id_data(id)->get_state().state() == active);
            HPX_ASSERT(state != active);
            statex = self_.yield(threads::thread_result_type(state,
                threads::invalid_thread_id));
            HPX_ASSERT(get_thread_id_data(id)->get_state().state() == active);
        }

        // handle interruption, if needed
        threads::interruption_point(id);

        // handle interrupt and abort
        if (statex == threads::wait_abort)
        {
            HPX_THROW_EXCEPTION(yield_aborted, desc,
                hpx::util::format(
                    "thread({}) aborted (yield returned wait_abort)",
                    description()));
        }

        return statex;
    }

    void execution_agent::do_resume(
        const char* desc, hpx::threads::thread_state_ex_enum statex)
    {
        threads::detail::set_thread_state(self_.get_thread_id(), pending,
            statex, thread_priority_normal, thread_schedule_hint{}, false);
    }
}}    // namespace hpx::threads
