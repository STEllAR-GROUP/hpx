//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/register_locks.hpp>
#if HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION
#include <hpx/util/backtrace.hpp>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    thread_state set_thread_state(thread_id_type const& id, thread_state_enum state,
        thread_state_ex_enum stateex, thread_priority priority, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::set_thread_state",
                "global applier object is not accessible");
            return thread_state(unknown);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().set_state(id, state, stateex,
            priority, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_id_type set_thread_state(thread_id_type const& id,
        boost::posix_time::ptime const& at_time, thread_state_enum state,
        thread_state_ex_enum stateex, thread_priority priority, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::set_thread_state",
                "global applier object is not accessible");
            return invalid_thread_id;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().set_state(at_time, id, state,
            stateex, priority, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_id_type set_thread_state(thread_id_type const& id,
        boost::posix_time::time_duration const& after, thread_state_enum state,
        thread_state_ex_enum stateex, thread_priority priority, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::set_thread_state",
                "global applier object is not accessible");
            return invalid_thread_id;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().set_state(after, id, state,
            stateex, priority, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_state get_thread_state(thread_id_type const& id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_state",
                "global applier object is not accessible");
            return thread_state(unknown);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_state(id);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_thread_phase(thread_id_type const& id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_phase",
                "global applier object is not accessible");
            return std::size_t(~0);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_phase(id);
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::thread_priority get_thread_priority(thread_id_type const& id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_priority",
                "global applier object is not accessible");
            return threads::thread_priority_unknown;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_priority(id);
    }

    std::ptrdiff_t get_stack_size(thread_id_type const& id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_stack_size",
                "global applier object is not accessible");
            return threads::thread_priority_unknown;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_stack_size(id);
    }

    void interrupt_thread(thread_id_type const& id, bool flag, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::interrupt_thread",
                "global applier object is not accessible");
            return;
        }
        app->get_thread_manager().interrupt(id, flag, ec);
    }

    void interruption_point(thread_id_type const& id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::interruption_point",
                "global applier object is not accessible");
            return;
        }
        app->get_thread_manager().interruption_point(id, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool get_thread_interruption_enabled(thread_id_type const& id,
        error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_interruption_enabled",
                "global applier object is not accessible");
            return false;
        }
        return app->get_thread_manager().get_interruption_enabled(id, ec);
    }

    bool set_thread_interruption_enabled(thread_id_type const& id, bool enable,
        error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_interruption_enabled",
                "global applier object is not accessible");
            return false;
        }
        return app->get_thread_manager().set_interruption_enabled(id, enable, ec);
    }

    bool get_thread_interruption_requested(thread_id_type const& id,
        error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_interruption_requested",
                "global applier object is not accessible");
            return false;
        }
        return app->get_thread_manager().get_interruption_requested(id, ec);
    }

#if HPX_THREAD_MAINTAIN_LOCAL_STORAGE
    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_thread_data(thread_id_type const& id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_data",
                "global applier object is not accessible");
            return 0;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_thread_data(id, ec);
    }

    std::size_t set_thread_data(thread_id_type const& id, std::size_t d,
        error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::set_thread_data",
                "global applier object is not accessible");
            return 0;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().set_thread_data(id, d, ec);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    void run_thread_exit_callbacks(thread_id_type const& id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::run_thread_exit_callbacks",
                "global applier object is not accessible");
            return;
        }
        app->get_thread_manager().run_thread_exit_callbacks(id, ec);
    }

    bool add_thread_exit_callback(thread_id_type const& id,
        HPX_STD_FUNCTION<void()> const& f, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::add_thread_exit_callback",
                "global applier object is not accessible");
            return false;
        }
        return app->get_thread_manager().add_thread_exit_callback(id, f, ec);
    }

    void free_thread_exit_callbacks(thread_id_type const& id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::add_thread_exit_callback",
                "global applier object is not accessible");
            return;
        }
        app->get_thread_manager().free_thread_exit_callbacks(id, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    char const* get_thread_description(thread_id_type const& id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_description",
                "global applier object is not accessible");
            return NULL;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_description(id);
    }
    char const* set_thread_description(thread_id_type const& id,
        char const* desc, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::set_thread_description",
                "global applier object is not accessible");
            return NULL;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().set_description(id, desc);
    }

    char const* get_thread_lco_description(thread_id_type const& id,
        error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_lco_description",
                "global applier object is not accessible");
            return NULL;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_lco_description(id);
    }
    char const* set_thread_lco_description(thread_id_type const& id,
        char const* desc, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::set_thread_lco_description",
                "global applier object is not accessible");
            return NULL;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().set_lco_description(id, desc);
    }

    ///////////////////////////////////////////////////////////////////////////
#if HPX_THREAD_MAINTAIN_FULLBACKTRACE_ON_SUSPENSION != 0
    char const* get_thread_backtrace(thread_id_type const& id, error_code& ec)
#else
    util::backtrace const* get_thread_backtrace(thread_id_type const& id, error_code& ec)
#endif
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_thread_backtrace",
                "global applier object is not accessible");
            return NULL;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().get_backtrace(id);
    }

#if HPX_THREAD_MAINTAIN_FULLBACKTRACE_ON_SUSPENSION != 0
    char const* set_thread_backtrace(thread_id_type const& id,
        char const* bt, error_code& ec)
#else
    util::backtrace const* set_thread_backtrace(thread_id_type const& id,
        util::backtrace const* bt, error_code& ec)
#endif
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::set_thread_backtrace",
                "global applier object is not accessible");
            return NULL;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return app->get_thread_manager().set_backtrace(id, bt);
    }

    threads::executor get_executor(thread_id_type const& id, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::threads::get_executor",
                "global applier object is not accessible");
            return default_executor();
        }

        return app->get_thread_manager().get_executor(id, ec);
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

#if HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION
        struct reset_backtrace
        {
            reset_backtrace(threads::thread_id_type const& id, error_code& ec)
              : id_(id),
                backtrace_(new hpx::util::backtrace()),
#if HPX_THREAD_MAINTAIN_FULLBACKTRACE_ON_SUSPENSION != 0
                full_backtrace_(backtrace_->trace()),
#endif
                ec_(ec)
            {
#if HPX_THREAD_MAINTAIN_FULLBACKTRACE_ON_SUSPENSION != 0
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
#if HPX_THREAD_MAINTAIN_FULLBACKTRACE_ON_SUSPENSION != 0
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
#if HPX_HAVE_VERIFY_LOCKS
            util::verify_no_locks();
#endif
#if HPX_THREAD_MAINTAIN_DESCRIPTION
            detail::reset_lco_description desc(id, description, ec);
#endif
#if HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION
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
            hpx::util::osstream strm;
            strm << "thread(" << threads::get_self_id() << ", "
                  << threads::get_thread_description(id)
                  << ") aborted (yield returned wait_abort)";
            HPX_THROWS_IF(ec, yield_aborted, description,
                hpx::util::osstream_get_string(strm));
        }

        if (&ec != &throws)
            ec = make_success_code();

        return statex;
    }

    threads::thread_state_ex_enum suspend(boost::posix_time::ptime const& at_time,
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
#if HPX_HAVE_VERIFY_LOCKS
            // verify that there are no more registered locks for this OS-thread
            util::verify_no_locks();
#endif
#if HPX_THREAD_MAINTAIN_DESCRIPTION
            detail::reset_lco_description desc(id, description, ec);
#endif
#if HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION
            detail::reset_backtrace bt(id, ec);
#endif
            threads::set_thread_state(id,
                at_time, threads::pending, threads::wait_signaled,
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
            hpx::util::osstream strm;
            strm << "thread(" << threads::get_self_id() << ", "
                  << threads::get_thread_description(id)
                  << ") aborted (yield returned wait_abort)";
            HPX_THROWS_IF(ec, yield_aborted, description,
                hpx::util::osstream_get_string(strm));
        }

        if (&ec != &throws)
            ec = make_success_code();

        return statex;
    }

    threads::thread_state_ex_enum suspend(
        boost::posix_time::time_duration const& after_duration,
        char const* description, error_code& ec)
    {
        // schedule a thread waking us up after_duration
        threads::thread_self& self = threads::get_self();
        threads::thread_id_type id = threads::get_self_id();

        // handle interruption, if needed
        threads::interruption_point(id, ec);
        if (ec) return threads::wait_unknown;

        // let the thread manager do other things while waiting
        threads::thread_state_ex_enum statex = threads::wait_unknown;

        {
#if HPX_HAVE_VERIFY_LOCKS
            // verify that there are no more registered locks for this OS-thread
            util::verify_no_locks();
#endif
#if HPX_THREAD_MAINTAIN_DESCRIPTION
            detail::reset_lco_description desc(id, description, ec);
#endif
#if HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION
            detail::reset_backtrace bt(id, ec);
#endif
            threads::set_thread_state(id,
                after_duration, threads::pending, threads::wait_signaled,
                threads::thread_priority_boost, ec);
            if (ec) return threads::wait_unknown;

            // suspend the HPX-thread
            statex = self.yield(threads::suspended);
        }

        // handle interruption, if needed
        threads::interruption_point(id);
        if (ec) return threads::wait_unknown;

        // handle interrupt and abort
        if (statex == threads::wait_abort) {
            hpx::util::osstream strm;
            strm << "thread(" << threads::get_self_id() << ", "
                  << threads::get_thread_description(id)
                  << ") aborted (yield returned wait_abort)";
            HPX_THROWS_IF(ec, yield_aborted, description,
                hpx::util::osstream_get_string(strm));
        }

        if (&ec != &throws)
            ec = make_success_code();

        return statex;
    }
}}

