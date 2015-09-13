//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/executors/generic_thread_pool_executor.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
#include <hpx/util/backtrace.hpp>
#endif

#include <sstream>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    thread_state set_thread_state(thread_id_type const& id, thread_state_enum state,
        thread_state_ex_enum stateex, thread_priority priority, error_code& ec)
    {
        if (&ec != &throws)
            ec = make_success_code();

        return  detail::set_thread_state(id, state, stateex,
            priority, std::size_t(-1), ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_id_type set_thread_state(thread_id_type const& id,
        util::steady_time_point const& abs_time, thread_state_enum state,
        thread_state_ex_enum stateex, thread_priority priority, error_code& ec)
    {
        return detail::set_thread_state_timed(*id->get_scheduler_base(), abs_time, id,
            state, stateex, priority, std::size_t(-1), ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_state get_thread_state(thread_id_type const& id, error_code& ec)
    {
        return id ? id->get_state() : thread_state(terminated);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_thread_phase(thread_id_type const& id, error_code& ec)
    {
        return id ? id->get_thread_phase() : std::size_t(~0);;
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
                "NULL thread id encountered");
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
                "NULL thread id encountered");
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
                "NULL thread id encountered");
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
                "NULL thread id encountered");
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
                "NULL thread id encountered");
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id->interruption_requested();
    }

#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_thread_data(thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::get_thread_data",
                "NULL thread id encountered");
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
                "NULL thread id encountered");
            return 0;
        }

        return id->set_thread_data(data);
    }
#endif

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

        if (0 == continuation_recursion_count.get())
            continuation_recursion_count.reset(new std::size_t(0));

        return *continuation_recursion_count.get();
    }

    ///////////////////////////////////////////////////////////////////////////
    void run_thread_exit_callbacks(thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::run_thread_exit_callbacks",
                "NULL thread id encountered");
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
                "NULL thread id encountered");
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
                "NULL thread id encountered");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        id->free_thread_exit_callbacks();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The get_thread_description function is part of the thread related API and
    /// allows to query the description of one of the thread id
    char const* get_thread_description(thread_id_type const& id, error_code& ec)
    {
        return id ? id->get_description() : "<unknown>";
    }

    char const* set_thread_description(thread_id_type const& id,
        char const* desc, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::set_thread_description",
                "NULL thread id encountered");
            return NULL;
        }
        if (&ec != &throws)
            ec = make_success_code();

        return id->set_description(desc);
    }

    char const* get_thread_lco_description(thread_id_type const& id,
        error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::get_thread_lco_description",
                "NULL thread id encountered");
            return NULL;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id ? id->get_lco_description() : "<unknown>";
    }
    char const* set_thread_lco_description(thread_id_type const& id,
        char const* desc, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::set_thread_lco_description",
                "NULL thread id encountered");
            return NULL;
        }

        if (&ec != &throws)
            ec = make_success_code();

        if (id)
            return id->set_lco_description(desc);
        return NULL;
    }

    ///////////////////////////////////////////////////////////////////////////
#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
    char const* get_thread_backtrace(thread_id_type const& id, error_code& ec)
#else
    util::backtrace const* get_thread_backtrace(thread_id_type const& id, error_code& ec)
#endif
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::get_thread_backtrace",
                "NULL thread id encountered");
            return NULL;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id ? id->get_backtrace() : 0;
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
                "NULL thread id encountered");
            return NULL;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return id ? id->set_backtrace(bt) : 0;
    }

    threads::executors::generic_thread_pool_executor
        get_executor(thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::get_executor",
                "NULL thread id encountered");
            return executors::generic_thread_pool_executor(0);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return executors::generic_thread_pool_executor(id->get_scheduler_base());
    }
}}

namespace hpx { namespace this_thread
{
    namespace detail
    {
        struct reset_lco_description
        {
            reset_lco_description(threads::thread_id_type const& id,
                    char const* description, error_code& ec)
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
            char const* old_desc_;
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
            boost::scoped_ptr<hpx::util::backtrace> backtrace_;
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
    threads::thread_state_ex_enum suspend(threads::thread_state_enum state,
        char const* description, error_code& ec)
    {
        // let the thread manager do other things while waiting
        threads::thread_self& self = threads::get_self();
        threads::thread_id_type id = threads::get_self_id();

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

            // suspend the HPX-thread
            statex = self.yield(state);
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
            HPX_THROWS_IF(ec, yield_aborted, description,
                strm.str());
        }

        if (&ec != &throws)
            ec = make_success_code();

        return statex;
    }

    threads::thread_state_ex_enum suspend(
        util::steady_time_point const& abs_time,
        char const* description, error_code& ec)
    {
        // schedule a thread waking us up at_time
        threads::thread_self& self = threads::get_self();
        threads::thread_id_type id = threads::get_self_id();

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
            threads::set_thread_state(id,
                abs_time, threads::pending, threads::wait_signaled,
                threads::thread_priority_boost, ec);
            if (ec) return threads::wait_unknown;

            // suspend the HPX-thread
            statex = self.yield(threads::suspended);
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
            HPX_THROWS_IF(ec, yield_aborted, description,
                strm.str());
        }

        if (&ec != &throws)
            ec = make_success_code();

        return statex;
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::executors::generic_thread_pool_executor
        get_executor(error_code& ec)
    {
        return threads::get_executor(threads::get_self_id(), ec);
    }
}}

