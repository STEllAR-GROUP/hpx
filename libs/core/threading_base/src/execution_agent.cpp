//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/execution_base/register_locks.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>

#include <hpx/threading_base/detail/reset_lco_description.hpp>
#include <hpx/threading_base/execution_agent.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_description.hpp>

#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
#include <hpx/debugging/backtrace.hpp>
#include <hpx/threading_base/detail/reset_backtrace.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

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
        do_yield(desc, hpx::threads::thread_schedule_state::pending);
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
            do_yield(desc, hpx::threads::thread_schedule_state::pending_boost);
        }
        else
        {
            do_yield(desc, hpx::threads::thread_schedule_state::pending);
        }
    }

    void execution_agent::resume(const char* desc)
    {
        do_resume(desc, threads::thread_restart_state::signaled);
    }

    void execution_agent::abort(const char* desc)
    {
        do_resume(desc, threads::thread_restart_state::abort);
    }

    void execution_agent::suspend(const char* desc)
    {
        do_yield(desc, threads::thread_schedule_state::suspended);
    }

    void execution_agent::sleep_for(
        hpx::chrono::steady_duration const& sleep_duration, const char* desc)
    {
        sleep_until(sleep_duration.from_now(), desc);
    }

    void execution_agent::sleep_until(
        hpx::chrono::steady_time_point const& sleep_time, const char* desc)
    {
        auto now = std::chrono::steady_clock::now();

        // Just yield until time has passed by...
        for (std::size_t k = 0; now < sleep_time.value(); ++k)
        {
            yield_k(k, desc);
            now = std::chrono::steady_clock::now();
        }
    }

#if defined(HPX_HAVE_VERIFY_LOCKS)
    struct on_exit_reset_held_lock_data
    {
        on_exit_reset_held_lock_data()
          : data_(hpx::util::get_held_locks_data())
        {
        }

        ~on_exit_reset_held_lock_data()
        {
            hpx::util::set_held_locks_data(std::move(data_));
        }

        std::unique_ptr<hpx::util::held_locks_data> data_;
    };
#else
    struct on_exit_reset_held_lock_data
    {
    };
#endif

    hpx::threads::thread_restart_state execution_agent::do_yield(
        const char* desc, threads::thread_schedule_state state)
    {
        thread_id_type id = self_.get_thread_id();
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROW_EXCEPTION(null_thread_id, "execution_agent::do_yield",
                "null thread id encountered (is this executed on a "
                "HPX-thread?)");
        }

        // handle interruption, if needed
        thread_data* thrd_data = get_thread_id_data(id);
        HPX_ASSERT(thrd_data);
        thrd_data->interruption_point();

        thrd_data->set_last_worker_thread_num(
            hpx::get_local_worker_thread_num());

        threads::thread_restart_state statex =
            threads::thread_restart_state::unknown;

        {
#ifdef HPX_HAVE_THREAD_DESCRIPTION
            threads::detail::reset_lco_description desc(
                id, util::thread_description(desc));
#endif
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
            threads::detail::reset_backtrace bt(id);
#endif
            on_exit_reset_held_lock_data held_locks;
            HPX_UNUSED(held_locks);

            HPX_ASSERT(thrd_data->get_state().state() ==
                thread_schedule_state::active);
            HPX_ASSERT(state != thread_schedule_state::active);
            statex = self_.yield(
                threads::thread_result_type(state, threads::invalid_thread_id));
            HPX_ASSERT(get_thread_id_data(id)->get_state().state() ==
                thread_schedule_state::active);
        }

        // handle interruption, if needed
        thrd_data->interruption_point();

        // handle interrupt and abort
        if (statex == threads::thread_restart_state::abort)
        {
            HPX_THROW_EXCEPTION(yield_aborted, desc,
                hpx::util::format(
                    "thread({}) aborted (yield returned wait_abort)",
                    description()));
        }

        return statex;
    }

    void execution_agent::do_resume(
        const char* /* desc */, hpx::threads::thread_restart_state statex)
    {
        thread_id_type id = self_.get_thread_id();

        thread_state previous_state;
        std::size_t k = 0;
        do
        {
            previous_state = get_thread_id_data(id)->get_state();
            thread_schedule_state previous_state_val = previous_state.state();

            // nothing to do here if the state doesn't change
            if (previous_state_val ==
                hpx::threads::thread_schedule_state::pending)
            {
                LTM_(warning)
                    << "resume: old thread state is already pending "
                       "thread state, aborting state change, thread("
                    << id << "), description("
                    << get_thread_id_data(id)->get_description() << ")";
                return;
            }
            switch (previous_state_val)
            {
            // The thread is still running... we yield our current context
            // and retry..
            case thread_schedule_state::active:
            {
                hpx::execution_base::this_thread::yield_k(
                    k, "hpx::threads::execution_agent::resume");
                ++k;
                LTM_(warning)
                    << "resume: thread is active, retrying state change, "
                       "thread("
                    << id << "), description("
                    << get_thread_id_data(id)->get_description() << ")";
                continue;
            }
            case thread_schedule_state::terminated:
            {
                LTM_(warning)
                    << "resume: thread is terminated, aborting state "
                       "change, thread("
                    << id << "), description("
                    << get_thread_id_data(id)->get_description() << ")";
                return;
            }
            case thread_schedule_state::pending:
            case thread_schedule_state::pending_boost:
            case thread_schedule_state::suspended:
                // We can now safely set the new state...
                break;
            case thread_schedule_state::pending_do_not_schedule:
            default:
            {
                // should not happen...
                std::ostringstream strm;
                strm << "resume: previous state was "
                     << get_thread_state_name(previous_state_val) << " ("
                     << previous_state_val << ")";
                HPX_ASSERT_MSG(false, strm.str().c_str());
                break;
            }
            }

            // If the previous state was pending we are supposed to remove the
            // thread from the queue. But in order to avoid linearly looking
            // through the queue we defer this to the thread function, which
            // at some point will ignore this thread by simply skipping it
            // (if it's not pending anymore).

            LTM_(info) << "resume: thread(" << id << "), description("
                       << get_thread_id_data(id)->get_description()
                       << "), old state("
                       << get_thread_state_name(previous_state_val) << ")";

            // So all what we do here is to set the new state.
            if (get_thread_id_data(id)->restore_state(
                    thread_schedule_state::pending, statex, previous_state))
            {
                break;
            }

            // state has changed since we fetched it from the thread, retry
            LTM_(error)
                << "resume: state has been changed since it was fetched, "
                   "retrying, thread("
                << id << "), description("
                << get_thread_id_data(id)->get_description() << "), old state("
                << get_thread_state_name(previous_state_val) << ")";

        } while (true);

        thread_schedule_state previous_state_val = previous_state.state();
        if (!(previous_state_val == thread_schedule_state::pending ||
                previous_state_val == thread_schedule_state::pending_boost))
        {
            auto* data = get_thread_id_data(id);
            auto scheduler = data->get_scheduler_base();
            auto hint = thread_schedule_hint(
                static_cast<std::int16_t>(data->get_last_worker_thread_num()));
            scheduler->schedule_thread(data, hint, true, data->get_priority());
            // Wake up scheduler
            scheduler->do_some_work(hint.hint);
        }
    }
}}    // namespace hpx::threads
