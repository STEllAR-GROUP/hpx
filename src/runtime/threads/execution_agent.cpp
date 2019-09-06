//  Copyright (c) 2019 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/format.hpp>
#include <hpx/logging.hpp>

#include <hpx/runtime/threads/execution_agent.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>

#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
#include <hpx/util/backtrace.hpp>
#endif

#include <hpx/util/thread_description.hpp>

#include <sstream>

namespace hpx { namespace threads {
    execution_agent::execution_agent(
        coroutines::detail::coroutine_impl* coroutine) noexcept
      : self_(coroutine)
    {
    }

    std::string execution_agent::description() const
    {
        thread_id_type id = self_.get_thread_id();
        return hpx::util::format("{}: {}", id, id->get_description());
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
            HPX_ASSERT(id->get_state().state() == active);
            HPX_ASSERT(state != active);
            statex = self_.yield(threads::thread_result_type(state, nullptr));
            HPX_ASSERT(id->get_state().state() == active);
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
        // HPX_ASSERT(statex == threads::wait_signaled);
    }

    void execution_agent::do_resume(
        const char* desc, hpx::threads::thread_state_ex_enum statex)
    {
        thread_id_type id = self_.get_thread_id();

        thread_state previous_state;
        std::size_t k = 0;
        do
        {
            previous_state = id->get_state();
            thread_state_enum previous_state_val = previous_state.state();

            // nothing to do here if the state doesn't change
            if (previous_state_val == hpx::threads::pending)
            {
                LTM_(warning)
                    << "resume: old thread state is already pending "
                       "thread state, aborting state change, thread("
                    << id << "), description(" << id->get_description() << ")";
                return;
            }
            switch (previous_state_val)
            {
            // The thread is still running... we yield our current context
            // and retry..
            case active: {
                hpx::basic_execution::this_thread::yield_k(
                    k, "hpx::threads::execution_agent::resume");
                ++k;
                LTM_(warning)
                    << "resume: thread is active, retrying state "
                       "change, thread("
                    << id << "), description(" << id->get_description() << ")";
                continue;
            }
            case terminated: {
                LTM_(warning)
                    << "resume: thread is terminated, aborting state "
                       "change, thread("
                    << id << "), description(" << id->get_description() << ")";
                return;
            }
            case pending:
            case pending_boost:
            case suspended:
                // We can now safely set the new state...
                break;
            case pending_do_not_schedule:
            default: {
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

            LTM_(info) << "resume: thread(" << id
                       << "), "
                          "description("
                       << id->get_description()
                       << "), "
                          "old state("
                       << get_thread_state_name(previous_state_val) << ")";

            // So all what we do here is to set the new state.
            if (id->restore_state(pending, statex, previous_state))
                break;

            // state has changed since we fetched it from the thread, retry
            LTM_(error)
                << "resume: state has been changed since it was fetched, "
                   "retrying, thread("
                << id
                << "), "
                   "description("
                << id->get_description()
                << "), "
                   "old state("
                << get_thread_state_name(previous_state_val) << ")";

        } while (true);

        thread_state_enum previous_state_val = previous_state.state();
        if (!(previous_state_val == pending ||
                previous_state_val == pending_boost))
        {
            auto scheduler = id->get_scheduler_base();
            auto hint = thread_schedule_hint();
            scheduler->schedule_thread(
                id.get(), thread_schedule_hint(), true, id->get_priority());
            // Wake up scheduler
            scheduler->do_some_work(hint.hint);
        }
    }
}}    // namespace hpx::threads
