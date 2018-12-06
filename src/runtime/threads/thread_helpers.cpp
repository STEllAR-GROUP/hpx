//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/threads/thread_helpers.hpp>

#include <hpx/error_code.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime.hpp>
#include <hpx/state.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/executors/current_executor.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_pool_base.hpp>
#include <hpx/util/assert.hpp>
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
#include <hpx/util/backtrace.hpp>
#endif
#ifdef HPX_HAVE_VERIFY_LOCKS
#  include <hpx/util/register_locks.hpp>
#endif
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/util/yield_while.hpp>

#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    thread_state set_thread_state(thread_id_type const& id, thread_state_enum state,
        thread_state_ex_enum stateex, thread_priority priority, error_code& ec)
    {
        if (&ec != &throws)
            ec = make_success_code();

        return detail::set_thread_state(id, state, stateex,
            priority, thread_schedule_hint(), ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_id_type set_thread_state(thread_id_type const& id,
        util::steady_time_point const& abs_time, std::atomic<bool>* timer_started,
        thread_state_enum state, thread_state_ex_enum stateex,
        thread_priority priority, error_code& ec)
    {
        return detail::set_thread_state_timed(*id->get_scheduler_base(), abs_time, id,
            state, stateex, priority, thread_schedule_hint(), timer_started, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_state get_thread_state(thread_id_type const& id, error_code& ec)
    {
        return id ? id->get_state() : thread_state(terminated, wait_unknown);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_thread_phase(thread_id_type const& id, error_code& ec)
    {
        return id ? id->get_thread_phase() : std::size_t(~0);;
    }

    std::size_t get_numa_node_number()
    {
        auto tid = get_self_id();
        auto pool = tid->get_scheduler_base()->get_parent_pool();
        auto num_thread = pool->get_worker_thread_num() + pool->get_thread_offset();
        auto pu_num = hpx::resource::get_partitioner().get_pu_num(num_thread);
        return hpx::threads::get_topology().get_numa_node_number(pu_num);
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::thread_priority get_thread_priority(thread_id_type const& id,
        error_code& ec)
    {
        return id ? id->get_priority() : thread_priority_unknown;
    }

    /// The get_stack_size function is part of the thread related API. It
    std::ptrdiff_t get_stack_size(thread_id_type const& id, error_code& ec)
    {
        return id ? id->get_stack_size() :
            static_cast<std::ptrdiff_t>(thread_stacksize_unknown);
    }

    void interrupt_thread(thread_id_type const& id, bool flag, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::interrupt_thread",
                "null thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        id->interrupt(flag);      // notify thread

        // set thread state to pending, if the thread is currently active,
        // this will be rescheduled until it calls an interruption point
        set_thread_state(id, pending, wait_abort,
            thread_priority_normal, ec);
    }

    void interruption_point(thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::interruption_point",
                "null thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        id->interruption_point();      // notify thread
    }

    ///////////////////////////////////////////////////////////////////////////
    bool get_thread_interruption_enabled(thread_id_type const& id,
        error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "hpx::threads::get_thread_interruption_enabled",
                "null thread id encountered");
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id->interruption_enabled();
    }

    bool set_thread_interruption_enabled(thread_id_type const& id, bool enable,
        error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "hpx::threads::get_thread_interruption_enabled",
                "null thread id encountered");
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id->set_interruption_enabled(enable);
    }

    bool get_thread_interruption_requested(thread_id_type const& id,
        error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::get_thread_interruption_requested",
                "null thread id encountered");
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id->interruption_requested();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_thread_data(thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::get_thread_data",
                "null thread id encountered");
            return 0;
        }

        return id->get_thread_data();
    }

    std::size_t set_thread_data(thread_id_type const& id, std::size_t data,
        error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::set_thread_data",
                "null thread id encountered");
            return 0;
        }

        return id->set_thread_data(data);
    }

    ////////////////////////////////////////////////////////////////////////////
    struct continuation_recursion_count_tag {};
    static util::thread_specific_ptr<
            std::size_t, continuation_recursion_count_tag
        > continuation_recursion_count;

    std::size_t& get_continuation_recursion_count()
    {
        thread_self* self_ptr = get_self_ptr();
        if (self_ptr)
            return self_ptr->get_continuation_recursion_count();

        if (nullptr == continuation_recursion_count.get())
            continuation_recursion_count.reset(new std::size_t(0));

        return *continuation_recursion_count.get();
    }

    void reset_continuation_recursion_count()
    {
        continuation_recursion_count.reset(nullptr);
    }

    ///////////////////////////////////////////////////////////////////////////
    void run_thread_exit_callbacks(thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::run_thread_exit_callbacks",
                "null thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        id->run_thread_exit_callbacks();
    }

    bool add_thread_exit_callback(thread_id_type const& id,
        util::function_nonser<void()> const& f, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::add_thread_exit_callback",
                "null thread id encountered");
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id->add_thread_exit_callback(f);
    }

    void free_thread_exit_callbacks(thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::add_thread_exit_callback",
                "null thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        id->free_thread_exit_callbacks();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The get_thread_description function is part of the thread related API and
    /// allows to query the description of one of the thread id
    util::thread_description get_thread_description(thread_id_type const& id,
        error_code& ec)
    {
        return id ? id->get_description() : util::thread_description("<unknown>");
    }

    util::thread_description set_thread_description(thread_id_type const& id,
        util::thread_description const& desc, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::set_thread_description",
                "null thread id encountered");
            return util::thread_description();
        }
        if (&ec != &throws)
            ec = make_success_code();

        return id->set_description(desc);
    }

    util::thread_description get_thread_lco_description(
        thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::get_thread_lco_description",
                "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id ? id->get_lco_description() : "<unknown>";
    }

    util::thread_description set_thread_lco_description(
        thread_id_type const& id, util::thread_description const& desc,
        error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::set_thread_lco_description",
                "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        if (id)
            return id->set_lco_description(desc);
        return nullptr;
    }

    ///////////////////////////////////////////////////////////////////////////
#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
    char const* get_thread_backtrace(thread_id_type const& id, error_code& ec)
#else
    util::backtrace const* get_thread_backtrace(thread_id_type const& id,
        error_code& ec)
#endif
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::get_thread_backtrace",
                "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id ? id->get_backtrace() : nullptr;
    }

#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
    char const* set_thread_backtrace(thread_id_type const& id,
        char const* bt, error_code& ec)
#else
    util::backtrace const* set_thread_backtrace(thread_id_type const& id,
        util::backtrace const* bt, error_code& ec)
#endif
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::set_thread_backtrace",
                "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id ? id->set_backtrace(bt) : nullptr;
    }

    threads::executors::current_executor
        get_executor(thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::get_executor",
                "null thread id encountered");
            return executors::current_executor(nullptr);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return executors::current_executor(id->get_scheduler_base());
    }

    threads::thread_pool_base*
        get_pool(thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::get_pool",
                "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id->get_scheduler_base()->get_parent_pool();
    }
}}

namespace hpx { namespace this_thread
{
    namespace detail
    {
        struct reset_lco_description
        {
            reset_lco_description(threads::thread_id_type const& id,
                    util::thread_description const& description,
                    error_code& ec)
              : id_(id), ec_(ec)
            {
                old_desc_ = threads::set_thread_lco_description(id_,
                    description, ec_);
            }

            ~reset_lco_description()
            {
                threads::set_thread_lco_description(id_, old_desc_, ec_);
            }

            threads::thread_id_type id_;
            util::thread_description old_desc_;
            error_code& ec_;
        };

#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
        struct reset_backtrace
        {
            reset_backtrace(threads::thread_id_type const& id, error_code& ec)
              : id_(id),
                backtrace_(new hpx::util::backtrace()),
#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
                full_backtrace_(backtrace_->trace()),
#endif
                ec_(ec)
            {
#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
                threads::set_thread_backtrace(id_, full_backtrace_.c_str(), ec_);
#else
                threads::set_thread_backtrace(id_, backtrace_.get(), ec_);
#endif
            }
            ~reset_backtrace()
            {
                threads::set_thread_backtrace(id_, 0, ec_);
            }

            threads::thread_id_type id_;
            std::unique_ptr<hpx::util::backtrace> backtrace_;
#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
            std::string full_backtrace_;
#endif
            error_code& ec_;
        };
#endif
    }

    /// The function \a suspend will return control to the thread manager
    /// (suspends the current thread). It sets the new state of this thread
    /// to the thread state passed as the parameter.
    ///
    /// If the suspension was aborted, this function will throw a
    /// \a yield_aborted exception.
    threads::thread_state_ex_enum suspend(
        threads::thread_state_enum state,
        threads::thread_id_type const& nextid,
        util::thread_description const& description, error_code& ec)
    {
        // let the thread manager do other things while waiting
        threads::thread_self& self = threads::get_self();
        threads::thread_id_type id = self.get_thread_id();

        // handle interruption, if needed
        threads::interruption_point(id, ec);
        if (ec) return threads::wait_unknown;

        threads::thread_state_ex_enum statex = threads::wait_unknown;

        {
            // verify that there are no more registered locks for this OS-thread
#ifdef HPX_HAVE_VERIFY_LOCKS
            util::verify_no_locks();
#endif
#ifdef HPX_HAVE_THREAD_DESCRIPTION
            detail::reset_lco_description desc(id, description, ec);
#endif
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
            detail::reset_backtrace bt(id, ec);
#endif
            // We might need to dispatch 'nextid' to it's correct scheduler
            // only if our current scheduler is the same, we should yield the id
            if (nextid && nextid->get_scheduler_base() != id->get_scheduler_base())
            {
                nextid->get_scheduler_base()->schedule_thread(
                    nextid.get(), threads::thread_schedule_hint());
                statex = self.yield(threads::thread_result_type(state,
                    threads::invalid_thread_id));
            }
            else
            {
                statex = self.yield(threads::thread_result_type(state, nextid));
            }
        }

        // handle interruption, if needed
        threads::interruption_point(id, ec);
        if (ec) return threads::wait_unknown;

        // handle interrupt and abort
        if (statex == threads::wait_abort)
        {
            std::ostringstream strm;
            strm << "thread(" << threads::get_self_id() << ", "
                  << threads::get_thread_description(id)
                  << ") aborted (yield returned wait_abort)";
            HPX_THROWS_IF(ec, yield_aborted, "suspend",
                strm.str());
        }

        if (&ec != &throws)
            ec = make_success_code();

        return statex;
    }

    threads::thread_state_ex_enum suspend(
        util::steady_time_point const& abs_time,
        threads::thread_id_type const& nextid,
        util::thread_description const& description, error_code& ec)
    {
        // schedule a thread waking us up at_time
        threads::thread_self& self = threads::get_self();
        threads::thread_id_type id = self.get_thread_id();

        // handle interruption, if needed
        threads::interruption_point(id, ec);
        if (ec) return threads::wait_unknown;

        // let the thread manager do other things while waiting
        threads::thread_state_ex_enum statex = threads::wait_unknown;

        {
#ifdef HPX_HAVE_VERIFY_LOCKS
            // verify that there are no more registered locks for this OS-thread
            util::verify_no_locks();
#endif
#ifdef HPX_HAVE_THREAD_DESCRIPTION
            detail::reset_lco_description desc(id, description, ec);
#endif
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
            detail::reset_backtrace bt(id, ec);
#endif
            std::atomic<bool> timer_started(false);
            threads::thread_id_type timer_id = threads::set_thread_state(id,
                abs_time, &timer_started, threads::pending, threads::wait_timeout,
                threads::thread_priority_boost, ec);
            if (ec) return threads::wait_unknown;

            // We might need to dispatch 'nextid' to it's correct scheduler
            // only if our current scheduler is the same, we should yield the id
            if (nextid && nextid->get_scheduler_base() != id->get_scheduler_base())
            {
                nextid->get_scheduler_base()->schedule_thread(
                    nextid.get(), threads::thread_schedule_hint());
                statex = self.yield(
                    threads::thread_result_type(threads::suspended,
                        threads::invalid_thread_id));
            }
            else
            {
                statex = self.yield(
                    threads::thread_result_type(threads::suspended, nextid));
            }

            if (statex != threads::wait_timeout)
            {
                HPX_ASSERT(
                    statex == threads::wait_abort ||
                    statex == threads::wait_signaled);
                error_code ec1(lightweight);    // do not throw
                hpx::util::yield_while(
                    [&timer_started]() { return !timer_started.load(); },
                    "set_thread_state_timed");
                threads::set_thread_state(timer_id,
                    threads::pending, threads::wait_abort,
                    threads::thread_priority_boost, ec1);
            }
        }

        // handle interruption, if needed
        threads::interruption_point(id, ec);
        if (ec) return threads::wait_unknown;

        // handle interrupt and abort
        if (statex == threads::wait_abort) {
            std::ostringstream strm;
            strm << "thread(" << threads::get_self_id() << ", "
                  << threads::get_thread_description(id)
                  << ") aborted (yield returned wait_abort)";
            HPX_THROWS_IF(ec, yield_aborted, "suspend_at",
                strm.str());
        }

        if (&ec != &throws)
            ec = make_success_code();

        return statex;
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::executors::current_executor get_executor(error_code& ec)
    {
        return threads::get_executor(threads::get_self_id(), ec);
    }

    threads::thread_pool_base* get_pool(error_code& ec)
    {
        return threads::get_pool(threads::get_self_id(), ec);
    }

    std::ptrdiff_t get_available_stack_space()
    {
        threads::thread_self *self = threads::get_self_ptr();
        if(self)
        {
            return self->get_available_stack_space();
        }

        return (std::numeric_limits<std::ptrdiff_t>::max)();
    }

    bool has_sufficient_stack_space(std::size_t space_needed)
    {
        if (nullptr == hpx::threads::get_self_ptr())
            return false;

#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
        std::ptrdiff_t remaining_stack = get_available_stack_space();
        if (remaining_stack < 0)
        {
            HPX_THROW_EXCEPTION(out_of_memory,
                "has_sufficient_stack_space", "Stack overflow");
        }
        bool sufficient_stack_space = std::size_t(remaining_stack) >= space_needed;

        // We might find ourselves in the situation where we don't have enough
        // stack space, but can't really schedule a new thread. In this sitation,
        // it would be best to change the code that provoked this behaviour
        // instead of dynamically schedule a new thread. A such, we throw an
        // exception to point to that problem instead of silently hanging because
        // the thread will never be executed.
        if (!sufficient_stack_space &&
            !hpx::threads::threadmanager_is(hpx::state::state_running))
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "has_sufficient_stack_space",
                "A potential stack overflow has been detected. Unable to "
                "schedule new thread during startup/shutdown.");
        }
        return sufficient_stack_space;
#else
        return true;
#endif
    }
}}
